import argparse
import asyncio
from pathlib import Path
from typing import List
from enum import Enum
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.vae_module.mld_vae import MldVaeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Request schema ---------
class EncodeRequest(BaseModel):
    name: str
    motion: List[List[float]]  # shape: [T, 263]

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# --------- Lifespan ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    latent_root = app.state.latent_root
    batch_size = app.state.batch_size
    batch_timeout = app.state.batch_timeout
    num_workers = app.state.num_workers

    logger.info("Starting background workers...")
    for worker_id in range(num_workers):
        asyncio.create_task(
            batch_worker_loop(
                worker_id,
                latent_root,
                batch_size=batch_size,
                timeout=batch_timeout,
            )
        )
    asyncio.create_task(queue_monitor_loop())
    logger.info(f"Started {num_workers} workers with batch_size={batch_size}")
    yield

# --------- App & queue ---------
app = FastAPI(lifespan=lifespan)
task_queue: asyncio.Queue = None
model: MldVaeModel = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 任务状态追踪
task_status_lock = asyncio.Lock()
task_status: dict[str, TaskStatus] = {}

# 队列使用率监控
queue_usage_threshold = 0.8  # 超过 80% 触发告警

# --------- Batch Worker ---------
async def batch_worker_loop(worker_id: int, latent_root: Path, batch_size: int = 8, timeout: float = 0.5):
    """
    批处理工作线程（支持多实例并发）：
    - 每个 worker 独立收集和处理批次
    - GPU 支持并发推理（CUDA 流）
    """
    logger.info(f"Worker {worker_id} started")
    
    while True:
        batch: List[EncodeRequest] = []
        
        # 收集批次
        start_time = asyncio.get_event_loop().time()
        while len(batch) < batch_size:
            try:
                timeout_remaining = timeout - (asyncio.get_event_loop().time() - start_time)
                if timeout_remaining <= 0:
                    break
                req = await asyncio.wait_for(task_queue.get(), timeout=timeout_remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break
        
        if not batch:
            await asyncio.sleep(0.1)
            continue
        
        logger.info(f"[Worker {worker_id}] Processing batch of {len(batch)} motions")
        
        # 批量编码
        try:
            motions_list = []
            lengths_list = []
            names_list = []
            
            # 准备数据
            for req in batch:
                motion_np = np.asarray(req.motion, dtype=np.float32)  # [T, 263]
                lengths_list.append(int(motion_np.shape[0]))
                motions_list.append(torch.tensor(motion_np).float())
                names_list.append(req.name)
            
            # 填充到相同长度
            dims = motions_list[0].dim()
            max_size = [max([motion.size(i) for motion in motions_list]) for i in range(dims)]
            size = (len(motions_list), ) + tuple(max_size)
            canvas = motions_list[0].new_zeros(size)
            for i, b in enumerate(motions_list):
                sub_tensor = canvas[i]
                for d in range(dims):
                    sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
                sub_tensor.add_(b)
            
            motion_tensor = canvas.to(device)
            
            # 批量编码（GPU 会自动排队多个 worker 的请求）
            with torch.no_grad():
                encoded = model.encode(
                    motion_tensor, 
                    lengths_list, 
                    return_dict=True
                ).latent
            
            latent_batch = encoded.permute(1, 0, 2).cpu().numpy() # [batch_size, token_len, latent_dim]
            
            # 保存每个样本
            latent_root.mkdir(parents=True, exist_ok=True)
            for idx, (name, latent_np) in enumerate(zip(names_list, latent_batch)):
                try:
                    latent_path = latent_root / f"{name}.npy"
                    np.save(latent_path, latent_np) # [1, 256]
                    
                    async with task_status_lock:
                        task_status[name] = TaskStatus.COMPLETED
                    logger.debug(f"[Worker {worker_id}] Saved latent for {name}")
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Failed to save {name}: {e}")
                    async with task_status_lock:
                        task_status[name] = TaskStatus.FAILED
        
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Batch encoding failed: {e}")
            async with task_status_lock:
                for name in names_list:
                    task_status[name] = TaskStatus.FAILED
        
        finally:
            for _ in batch:
                task_queue.task_done()

# --------- Queue Monitor ---------
async def queue_monitor_loop():
    """监控队列使用率"""
    while True:
        await asyncio.sleep(5)
        qsize = task_queue.qsize()
        maxsize = task_queue._maxsize
        usage = qsize / maxsize
        
        if usage > queue_usage_threshold:
            logger.warning(f"⚠️  Queue usage high: {qsize}/{maxsize} ({usage:.1%})")
        else:
            logger.info(f"Queue usage: {qsize}/{maxsize} ({usage:.1%})")

# --------- Routes ---------
@app.post("/encode")
async def enqueue_encode(req: EncodeRequest):
    latent_root = app.state.latent_root
    latent_path = latent_root / f"{req.name}.npy"
    
    async with task_status_lock:
        # 1. 检查文件是否已存在
        if latent_path.exists():
            return {"status": "exists", "message": f"Latent file already exists for {req.name}"}
        
        # 2. 检查任务是否在处理中
        current_status = task_status.get(req.name)
        if current_status == TaskStatus.PROCESSING:
            return {"status": "processing", "message": f"Motion {req.name} is being processed"}
        elif current_status == TaskStatus.COMPLETED:
            return {"status": "exists", "message": f"Motion {req.name} completed recently"}
        elif current_status == TaskStatus.FAILED:
            logger.info(f"Retrying previously failed task: {req.name}")
            task_status.pop(req.name, None)
        
        # 3. 检查队列是否满
        if task_queue.full():
            raise HTTPException(status_code=429, detail="Queue is full, retry later.")
        
        # 4. 入队并标记为处理中
        await task_queue.put(req)
        task_status[req.name] = TaskStatus.PROCESSING
    
    return {"status": "queued", "message": f"Motion {req.name} queued for encoding"}

@app.get("/task_status/{task_name}")
async def get_task_status(task_name: str):
    """查询某个任务的状态"""
    async with task_status_lock:
        status = task_status.get(task_name, "unknown")
    return {"task_name": task_name, "status": status}

@app.get("/queue_info")
async def get_queue_info():
    """获取队列信息"""
    async with task_status_lock:
        processing_count = sum(1 for s in task_status.values() if s == TaskStatus.PROCESSING)
        completed_count = sum(1 for s in task_status.values() if s == TaskStatus.COMPLETED)
    
    return {
        "queue_size": task_queue.qsize(),
        "queue_max_size": task_queue._maxsize,
        "queue_usage": f"{task_queue.qsize() / task_queue._maxsize:.1%}",
        "processing_tasks": processing_count,
        "completed_tasks": completed_count,
    }

# --------- Entry ---------
def main():
    parser = argparse.ArgumentParser(description="VAE Server for Motion Feature Encoding/Decoding")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained VAE model.')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address for the server.')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the server.')
    parser.add_argument('--queue_size', type=int, default=0, help='Max queue size (0=auto: 2*workers*batch_size).')
    parser.add_argument('--latent_dir', type=str, default='data/HumanML3D/latents', help='Where to save latent npy files.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoding.')
    parser.add_argument('--batch_timeout', type=float, default=0.5, help='Timeout for collecting batch (seconds).')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of concurrent batch workers')
    args = parser.parse_args()

    # 自动计算队列大小
    if args.queue_size == 0:
        args.queue_size = args.num_workers * args.batch_size * 2
        logger.info(f"Auto-calculated queue_size: {args.queue_size} "
                   f"(workers={args.num_workers}, batch_size={args.batch_size}, buffer=2x)")
    
    # 验证队列大小合理性
    min_queue_size = args.num_workers * args.batch_size * 2
    if args.queue_size < min_queue_size:
        logger.warning(f"⚠️  queue_size ({args.queue_size}) < recommended minimum ({min_queue_size})")

    global task_queue, model, device
    task_queue = asyncio.Queue(maxsize=args.queue_size)

    logger.info(f"Loading model from {args.model_path}...")
    model = MldVaeModel.from_pretrained_ckpt(args.model_path)
    model.eval()
    model.to(device)
    logger.info(f"Model loaded on {device}")

    latent_root = Path(args.latent_dir)
    
    # 将配置保存到 app.state（供 startup_event 使用）
    app.state.latent_root = latent_root
    app.state.batch_size = args.batch_size
    app.state.batch_timeout = args.batch_timeout
    app.state.num_workers = args.num_workers

    logger.info(f"Starting server: {args.num_workers} workers, batch_size={args.batch_size}, "
               f"queue_size={args.queue_size} on {args.host}:{args.port}")
    
    # 使用 uvicorn 的异步模式
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()