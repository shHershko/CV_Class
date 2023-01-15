"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        padding = int(win_size//2)
        padded_left = np.pad(left_image,((padding, padding), (dsp_range + padding, dsp_range + padding), (0, 0)),'constant', constant_values=(0))
        padded_right = np.pad(right_image,((padding,padding),(dsp_range+padding,dsp_range+padding),(0,0)),'constant', constant_values=(0))
        for i_row in np.arange(padding,num_of_rows+padding):
            for i_col in np.arange(padding+dsp_range,padding+dsp_range+num_of_cols):
                left_win = padded_left[i_row - padding: i_row + padding + 1, i_col - padding: i_col + padding + 1]
                for dispa_inx, dispa in enumerate(disparity_values):
                    right_win = padded_right[i_row-padding:i_row+padding+1,i_col+dispa-padding:i_col+dispa+padding+1]
                    ssdd_tensor[i_row-padding, i_col-dsp_range-padding, dispa_inx] = np.sum((left_win-right_win)**2)
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        label_no_smooth = np.argmin(ssdd_tensor, 2)  # Inx of the minimum value along the disparity
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        l_slice[:, 0] = c_slice[:, 0]
        mesh_inx = np.linspace(0, num_labels - 1, num_labels)
        w_mat, h_mat = np.meshgrid(mesh_inx, mesh_inx)
        labels_inx = np.abs(w_mat-h_mat)

        for col_inx in range(1, num_of_cols):
            M = np.array([l_slice[:, col_inx-1]] * num_labels) # a
            M[labels_inx == 1] = M[labels_inx == 1] + p1 # b
            M[labels_inx > 1] = M[labels_inx > 1] + p2 # c
            M = np.min(M, axis=1)

            l_slice[:, col_inx] = c_slice[:, col_inx] + M - np.min(l_slice[:, col_inx-1])

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        for i_row in range(ssdd_tensor.shape[0]):
            ssdd_slice = ssdd_tensor[i_row, :, :].T
            l[i_row, :, :] = (Solution.dp_grade_slice(ssdd_slice, p1, p2)).T
        return self.naive_labeling(l)

    def slice_ext(ssd_tensor: np.ndarray, direction: int, iter: int) -> np.ndarray:
        """extracts slices from the ssdd according to a direction which it recieves.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: one of the 8 directions defined for the exercise (assume only 1-8!)
            offset: the offset for the main direction (row, column or iter)
        Returns:
            the current slice from the ssdd tensor
        """

        if (direction == 1 or direction == 5):
            ssdd_slice = ssd_tensor[iter, :, :]
        elif (direction == 2 or direction == 6):
            ssdd_slice = np.diagonal(ssd_tensor, iter)
        elif (direction == 3 or direction == 7):
            ssdd_slice = ssd_tensor[:, iter, :]
        elif (direction == 4 or direction == 8):
            ssdd_slice = np.diagonal(np.fliplr(ssd_tensor), iter)

        if (direction <= 4):
            return ssdd_slice
        else:
            return np.flipud(ssdd_slice)

    def dp_grade_slice_dir(c_slice: np.ndarray, p1: float, p2: float,
                               direction: int) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.
           this is dp_grade_slice updated  to support diagonal directions (slices of shorter lengths)
        """
        if (direction %2 ==0):  
            return Solution.dp_grade_slice(c_slice.T, p1, p2).T   
        else: # not diagonal 
            return Solution.dp_grade_slice(c_slice, p1, p2)

    def l_indices_update(ssdd_tensor: np.ndarray,
            direction: int, indx: int, p1: float, p2: float):
        return Solution.dp_grade_slice_dir(Solution.slice_ext(ssdd_tensor, direction, indx).T, p1, p2, direction).T

    def dp_grade_per_direction(
            ssdd_tensor: np.ndarray,
            direction: int,
            p1: float,
            p2: float) -> np.ndarray:
        """l scores tensor for a direction
           each slice in direction corresponding l tensor

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: one of the 8 directions defined for the exercise (assume only 1-8!)
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            np.ndarray which maps each direction to the corresponding l score tensor
        """
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        num_rows, num_cols, num_dsp = ssdd_tensor.shape[0], ssdd_tensor.shape[1], ssdd_tensor.shape[2]
        ssdd_indices = np.arange(num_rows * num_cols * num_dsp)
        ssdd_indices = ssdd_indices.reshape((num_rows, num_cols, num_dsp))
        # for each direction we want to extract indices and values of ssdd
        # dp on them and assign to l according to the relevant indices
        if (direction % 2 == 0):
            for indx in range(-num_rows + 1, num_cols):
                indices_slice = Solution.slice_ext(ssdd_indices, direction, indx)
                indices = np.unravel_index(indices_slice, ssdd_tensor.shape)
                l[indices] = Solution.l_indices_update(ssdd_tensor,direction,indx,p1,p2)

        elif (direction == 1):
            # indx = row
            for indx in range(num_rows):
                l[indx, :, :] = Solution.l_indices_update(ssdd_tensor,direction,indx,p1,p2)
        elif (direction == 3):
            # indx = col
            for indx in range(num_cols):
                l[:, indx, :] = Solution.l_indices_update(ssdd_tensor,direction,indx,p1,p2)

        elif (direction == 5):
            # indx = row
            for indx in range(num_rows - 1, 0, -1):
                indices_slice = Solution.slice_ext(ssdd_indices, direction, indx)
                indices = np.unravel_index(indices_slice, ssdd_tensor.shape)
                l[indices] = Solution.l_indices_update(ssdd_tensor,direction,indx,p1,p2)
        else:
            # indx = col
            for indx in range(num_cols - 1, 0, -1):
                indices_slice = Solution.slice_ext(ssdd_indices, direction, indx)
                indices = np.unravel_index(indices_slice, ssdd_tensor.shape)
                l[indices] =Solution.l_indices_update(ssdd_tensor,direction,indx,p1,p2)
        return l

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        for direction in range(1,num_of_directions+1):
            l = Solution.dp_grade_per_direction(ssdd_tensor,direction,p1,p2)
            direction_to_slice[direction] = Solution.naive_labeling(l)
            l = np.zeros_like(ssdd_tensor)
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        for direction in range(1,num_of_directions+1):
            l += Solution.dp_grade_per_direction(ssdd_tensor,direction,p1,p2)
        l = l/num_of_directions # avg of all
        return self.naive_labeling(l)

