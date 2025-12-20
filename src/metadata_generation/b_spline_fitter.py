import numpy as np
from scipy.interpolate import BSpline, splrep

# SMPL 23 joint names
DEFAULT_JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"]


class BSplineFitter:
    """Base class for fitting B-Spline curves to data."""
    def _normalize_time(self, time_points: np.ndarray) -> np.ndarray:
        """Normalizes time points to the range [0, 1]."""
        assert time_points.max() > time_points.min(), "Time points must have a range greater than zero."
        return (time_points - time_points.min()) / (time_points.max() - time_points.min())
    
    def fit(
        self,
        time_points: np.ndarray,
        data_points: np.ndarray,
        degree: int = 3,
        control_points: int = 4,
        smoothness: float = 0.0,
        decimals: int = 2,
    ):
        """
        Fits a B-Spline to the given data points.
        
        Args:
            time_points (np.ndarray): 1D array of time points.
            data_points (np.ndarray): 2D array of data points corresponding to the time points.
            degree (int): Degree of the B-Spline.
            control_points (int): Number of control points for the spline.
            smoothness (float): Smoothing factor for the spline fitting.
            decimals (int): Number of decimal places to round the coefficients.
        Returns:
            List[BSpline]: List of B-Spline objects for each dimension of the data points.
        """
        # 1. Normalize time points to [0, 1]
        t_norm = self._normalize_time(time_points)
        # 2. Compute knot vector
        num_internal_knots = control_points - degree - 1
        if num_internal_knots <= 0:
            internal_knots = np.array([]) # No internal knots, splrep will auto choose knots' position.
        else:
            internal_knots = np.percentile(t_norm, np.linspace(0, 100, num_internal_knots + 2)[1:-1])
            
        # 3. Fit B-Spline for each dimension
        splines = []
        for dim in range(data_points.shape[1]):
            tck = splrep(t_norm, data_points[:, dim], k=degree, t=internal_knots, s=smoothness)
            coefficients = tck[1]
            if decimals is not None:
                coefficients = np.around(tck[1], decimals=decimals)
            splines.append(BSpline(tck[0], coefficients, tck[2]))
        
        return splines
    
    @staticmethod
    def get_spline_vaild_coefficients(
        spline: BSpline, 
        degree: int = 3,
        decimals: int = 2,
    ) -> np.ndarray:
        """
        Extracts and rounds the valid coefficients of the B-Spline.
        
        Args:
            spline (BSpline): The B-Spline object.
            degree (int): Degree of the B-Spline.
            decimals (int): Number of decimal places to round the coefficients.
        """
        assert spline.k == degree, "Spline degree does not match."
        # exclude order=degree+1 zero-padding coefficients
        order = degree + 1
        control_points = len(spline.c) - order
        coefficients = spline.c[:control_points]
        coefficients = np.around(coefficients, decimals=decimals)
        # Set -0.0 to 0 for cleaner output
        coefficients[np.abs(coefficients) == 0] = 0
        return coefficients
    
    @staticmethod
    def get_spline_from_coefficients(
        coefficients: np.ndarray,
        degree: int = 3,
    ) -> BSpline:
        """
        Constructs a B-Spline from the given coefficients.
        
        Args:
            coefficients (np.ndarray): Array of control point coefficients.
            degree (int): Degree of the B-Spline.
        """
        order = degree + 1
        # Create knot vector with clamped knots
        num_control_points = len(coefficients)
        num_knots = num_control_points + order
        knots = np.zeros(num_knots)
        # uniformly spaced internal knots
        knots[degree:num_knots - degree] = np.linspace(0, 1, num_knots - 2 * degree)
        knots[num_knots - degree:] = 1.0
        # Pad coefficients with zeros
        coefficients = np.concatenate([coefficients, np.zeros(order)])
        
        return BSpline(knots, coefficients, degree)


class BodyJointBSplineFitter(BSplineFitter):
    """Fits B-Spline curves to Body Joint sequences."""
    
    def __init__(self, degree: int = 3, num_control_points: int = 10):
        """
        Initializes the B-Spline fitter.
        
        Args:
            degree (int): Degree of the B-Spline.
            num_control_points (int): Number of control points for the spline.
        """
        self.degree = degree
        self.num_control_points = num_control_points
        assert self.num_control_points > self.degree, "Number of control points must be greater than the degree."
        self.splines = {}

    def fit_joint(self, joint_name: str, joint_data: np.ndarray):
        """
        Fits a B-Spline to the given joint data.
        
        Args:
            joint_name (str): Name of the joint.
            joint_data (np.ndarray): Array of shape (T, 3) representing the joint's parameters over T time frames.
        """
        pass
    
    def fit_all_joints(self, body_joint_data: dict[str, np.ndarray] | np.ndarray):
        """
        Fits B-Splines to all joints in the body joint data.
        
        Args:
            body_joint_data (dict[str, np.ndarray] | np.ndarray):
                If dict: mapping from joint names to their data arrays of shape (T, 3).
                If ndarray: array of shape (J, T, 3) where J is the number of joints.
        """
        if isinstance(body_joint_data, np.ndarray):
            for joint_name, joint_data in zip(DEFAULT_JOINT_NAMES, body_joint_data):
                self.fit_joint(joint_name, joint_data)
        else:
            for joint_name, joint_data in body_joint_data.items():
                self.fit_joint(joint_name, joint_data)
    
    def predict(self, joint_idx: int|str, t: int|list[int]) -> np.ndarray:
        """
        Predicts joint parameters at given time point using the fitted B-Spline.
        
        Args:
            joint_idx (int|str): Index or name of the joint.
            t (int|list[int]): Time point(s) at which to predict the joint parameters.
        """
        if isinstance(joint_idx, int):
            joint_name = list(self.splines.keys())[joint_idx]
        else:
            joint_name = joint_idx
        
        if isinstance(t, int):
            t = [t]
        else:
            t = self._normalize_time(np.array(t))

        spline = self.splines.get(joint_name)
        if spline is None:
            raise ValueError(f"No spline found for joint: {joint_name}")
        
        predicted_value = np.array([s(t) for s in spline]).T  # Shape (len(t), 3)
        
        return predicted_value
    
    def predict_all_joints(self, t: int|list[int], return_dict: bool = False) -> dict[str, np.ndarray] | np.ndarray:
        """
        Predicts all joint parameters at given time point(s) using the fitted B-Splines.
        
        Args:
            t (int|list[int]): Time point(s) at which to predict the joint parameters.
            return_dict (bool): If True, returns a dict mapping joint names to predicted values.
                                If False, returns an ndarray of shape (J, 3) where J is the number of joints.
        """
        predictions = {}
        for joint_name in self.splines.keys():
            predictions[joint_name] = self.predict(joint_name, t)
        
        if return_dict:
            return predictions
        else:
            joint_order = list(self.splines.keys())
            return np.array([predictions[joint_name] for joint_name in joint_order])


class CorrdinateBSplineFitter(BodyJointBSplineFitter):
    """Fits B-Spline curves to 3D coordinate sequences."""

    def fit_joint(self, joint_name: str, joint_data: np.ndarray):
        """
        Fits a B-Spline to the given joint coordinate data.
        
        Args:
            joint_name (str): Name of the joint.
            joint_data (np.ndarray): Array of shape (T, 3) representing the joint's 3D coordinates over T time frames.
        """
        T = joint_data.shape[0]
        t = np.arange(T)
        
        self.splines[joint_name] = self.fit(
            time_points=t,
            data_points=joint_data,
            degree=self.degree,
            control_points=self.num_control_points,
        )