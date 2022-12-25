"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        N = match_p_src.shape[1]
        u = match_p_dst[0, :]
        v = match_p_dst[1, :]

        # Transformation of the source cords to homogeneous coordinates
        xy_hom = np.transpose(np.vstack([match_p_src, np.ones(N)]))

        u_xy_hom = np.multiply(np.transpose(np.array([u, ] * 3)), xy_hom)
        v_xy_hom = np.multiply(np.transpose(np.array([v, ] * 3)), xy_hom)
        A_u_mat = np.concatenate((-xy_hom, np.zeros((N, 3)), u_xy_hom), axis=1)
        A_v_mat = np.concatenate((np.zeros((N, 3)), -xy_hom, v_xy_hom), axis=1)
        A_u_doubled = np.zeros((2 * N, 9))
        A_v_doubled = np.zeros((2 * N, 9))
        A_mat = np.zeros((2 * N, 9))
        for i in range(0, N):
            A_mat[2 * i] = A_u_mat[i]
            A_mat[2 * i + 1] = A_v_mat[i]
        A_svd_mat = np.dot(np.transpose(A_mat), A_mat)
        _, eig_v = np.linalg.eigh(A_svd_mat)
        H_mat = eig_v[:, 0]
        H_mat = np.reshape(H_mat, (3, 3))
        # H_mat = H_mat / np.max(np.abs(H_mat))
        return H_mat
        pass

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        image_for_calc = np.asarray(src_image[:, :, 1])
        y_vect, x_vect = np.where(image_for_calc!=None)
        x_vect = np.asarray(x_vect)
        y_vect = np.asarray(y_vect)
        x_vect = x_vect.T
        y_vect = y_vect.T
        src_ind = np.vstack((x_vect, y_vect,np.ones(x_vect.shape[0])))
        src2dest_ind_raw = homography @ src_ind
        src2dest_ind_raw = src2dest_ind_raw/src2dest_ind_raw[2,:]
        src2dest_ind_raw = src2dest_ind_raw[0:2,:]
        src2dest_ind_raw = src2dest_ind_raw.round()
        src2dest_ind_raw = src2dest_ind_raw.astype(int)
        src2dest_image = (np.zeros(dst_image_shape)).astype(np.uint8)
        src_ind = src_ind.astype(int)
        for i_pixel in range(src2dest_ind_raw.shape[1]):
           if (src2dest_ind_raw[0,i_pixel]< dst_image_shape[1]) & (src2dest_ind_raw[1,i_pixel]< dst_image_shape[0]) & (src2dest_ind_raw[0,i_pixel]>=0) & (src2dest_ind_raw[1,i_pixel]>=0):
               src2dest_image[src2dest_ind_raw[1,i_pixel],src2dest_ind_raw[0,i_pixel],:] = src_image[src_ind[1,i_pixel],src_ind[0,i_pixel],:]

               # print(src_image[src_ind[1,i_pixel],src_ind[0,i_pixel],:])
               # print(src2dest_image[src2dest_ind_raw[1,i_pixel],src2dest_ind_raw[0,i_pixel],:])
        return src2dest_image
        pass

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        x_vect = np.linspace(0, src_image.shape[1]-1, src_image.shape[1]).astype(int)
        y_vect = np.linspace(0, src_image.shape[0]-1, src_image.shape[0]).astype(int)
        x_mesh, y_mesh = np.meshgrid(x_vect,y_vect)
        x_reshaped = np.reshape(x_mesh,[src_image.shape[0]*src_image.shape[1], 1]).T
        y_reshaped = np.reshape(y_mesh, [src_image.shape[0] * src_image.shape[1], 1]).T
        color_mesh = np.reshape(src_image, [src_image.shape[0] * src_image.shape[1], 3]).T

        src_ind = np.vstack((x_reshaped, y_reshaped, np.ones(x_reshaped.shape[1])))
        src2dest_ind_raw = homography @ src_ind
        src2dest_ind_raw = src2dest_ind_raw/src2dest_ind_raw[2,:]
        src2dest_ind_raw = src2dest_ind_raw[0:2,:]
        src2dest_ind_raw = src2dest_ind_raw.round()
        src2dest_ind_raw = src2dest_ind_raw.astype(int)

        src2dest_image = (np.zeros(dst_image_shape)).astype(np.uint8)
        relevant_ind = src2dest_ind_raw >= 0
        relevant_ind[0, :] = relevant_ind[0, :] & (src2dest_ind_raw[0, :] < dst_image_shape[1])
        relevant_ind[1, :] = relevant_ind[1, :] & (src2dest_ind_raw[1, :] < dst_image_shape[0])
        relevant_ind = relevant_ind[0, :] & relevant_ind[1, :]
        src2dest_ind_filt = (src2dest_ind_raw[:, relevant_ind]).astype(int)
        mat_ind = np.repeat(relevant_ind, 3)
        mat_ind = np.reshape(mat_ind.T, [3, relevant_ind.shape[0]])
        src2dest_image[src2dest_ind_filt[1, :], src2dest_ind_filt[0, :],:] = color_mesh[:,relevant_ind].T


        return src2dest_image


        pass

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        match_src_mat = np.vstack((match_p_src, np.ones(match_p_src.shape[1]))).astype(int)
        match_dest_mat = np.vstack((match_p_dst, np.ones(match_p_dst.shape[1]))).astype(int)
        hom_trans_src = homography @ match_src_mat
        hom_trans_src = hom_trans_src/hom_trans_src[2]

        inliers_inx = []
        norms_vect  = []
        for point_inx in range(len(match_dest_mat[1])):
            curr_point_norm = np.linalg.norm(match_dest_mat[:,point_inx] - hom_trans_src[:,point_inx])
            if curr_point_norm < max_err:
                inliers_inx.append(point_inx)
                norms_vect.append(curr_point_norm)
        if not inliers_inx:
            dist_mse = 10 ** 9
        else:
            dist_mse = np.mean(norms_vect)
        fit_percent = len(inliers_inx)/match_p_src.shape[1]
        return fit_percent, dist_mse
        pass

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        match_src_mat = np.vstack((match_p_src, np.ones(match_p_src.shape[1]))).astype(int)
        match_dest_mat = np.vstack((match_p_dst, np.ones(match_p_dst.shape[1]))).astype(int)
        hom_trans_src = homography @ match_src_mat
        hom_trans_src = hom_trans_src/hom_trans_src[2]

        inliers_inx = []
        for point_inx in range(len(match_dest_mat[1])):
            curr_point_norm = np.linalg.norm(match_dest_mat[:,point_inx] - hom_trans_src[:,point_inx])
            if curr_point_norm < max_err:
                inliers_inx.append(point_inx)
            match_src_comp = match_p_src[:, inliers_inx]
            match_dest_comp = match_p_dst[:, inliers_inx]
        return match_src_comp, match_dest_comp

        pass

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        curr_best_mse = 10**10 # Only for initial value or no equal points
        iters_factor = 5
        n_comp_points = 6
        for iter_inx in range(iters_factor * k):
            chosen_points_inx = sample(range(match_p_dst.shape[1]), n_comp_points)
            curr_homography = Solution.compute_homography_naive(match_p_src[:, chosen_points_inx], match_p_dst[:, chosen_points_inx])
            curr_src_mm, curr_dest_mm = Solution.meet_the_model_points(curr_homography, match_p_src, match_p_dst, max_err)
            fit_percent, dist_mse = Solution.test_homography(curr_homography, match_p_src, match_p_dst, max_err)

            if fit_percent >= inliers_percent:
                curr_homography = Solution.compute_homography_naive(curr_src_mm, curr_dest_mm)
                fit_percent, dist_mse = Solution.test_homography(curr_homography, match_p_src, match_p_dst, max_err)
                if dist_mse < curr_best_mse:
                    homography = curr_homography
                    curr_best_mse = dist_mse

        return homography
        """INSERT YOUR CODE HERE"""
        pass

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        pass

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        pass
