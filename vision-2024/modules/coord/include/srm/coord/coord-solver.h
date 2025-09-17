#ifndef SRM_COORD_SOLVER_H_
#define SRM_COORD_SOLVER_H_

#include <srm/coord/coord-base.h>

#include <Eigen/LU>
#include <opencv2/core/mat.hpp>

namespace srm::coord {

/**
 * @brief 坐标系求解器类
 * 世界坐标系：原点为陀螺仪中心，自身无旋转，方向为陀螺仪置零的方向
 * 陀螺仪坐标系：原点为陀螺仪中心，自身可旋转
 * 相机坐标系：原点为相机光心，自身可旋转，与世界坐标系原点位置关系固定
 * 枪口坐标系：原点为子弹获得初速的位置，自身可旋转，与世界坐标系原点位置关系固定
 */
class Solver {
 private:
  cv::Mat intrinsic_mat_;                ///< 相机矩阵
  cv::Mat distortion_mat_;               ///< 畸变矩阵
  Eigen::Matrix3d intrinsic_mat_eigen_;  ///< Eigen格式相机矩阵
 public:
  [[nodiscard]] const Eigen::Matrix3d& intrinsic_mat_eigen() const { return intrinsic_mat_eigen_; }

 private:
  CTVec ctv_iw_;  ///< 陀螺仪相对世界坐标系原点的位移
  CTVec ctv_ci_;  ///< 相机相对陀螺仪的位移
  CTVec ctv_mi_;  ///< 枪口相对陀螺仪的位移
  CTVec ctv_cw_;  ///< 相机相对世界坐标系原点的位移
  CTVec ctv_mw_;  ///< 枪口相对世界坐标系原点的位移
  CTVec ctv_mc_;  ///< 枪口相对相机的位移
  RMat rm_ci_;    ///< 相机相对陀螺仪的旋转矩阵
  RMat rm_mi_;    ///< 枪口相对陀螺仪的旋转矩阵
  HTMat htm_ic_;  ///< 陀螺仪坐标系转换到相机坐标系
  HTMat htm_ci_;  ///< 相机坐标系转换到陀螺仪坐标系
  HTMat htm_im_;  ///< 陀螺仪坐标系转换到枪口坐标系
  HTMat htm_mi_;  ///< 枪口坐标系转换到陀螺仪坐标系

 public:
  attr_writer_val(intrinsic_mat_, InitIntrinsicMat);    ///< 设定相机矩阵
  attr_writer_val(distortion_mat_, InitDistortionMat);  ///< 设定畸变矩阵

  attr_reader_ref(ctv_iw_, CTVecIMUWorld);          ///< 陀螺仪相对世界坐标系原点的位移
  attr_reader_ref(ctv_ci_, CTVecCamIMU);            ///< 相机相对陀螺仪的位移
  attr_reader_ref(ctv_mi_, CTVecMuzzleIMU);         ///< 枪口相对陀螺仪的位移
  attr_reader_ref(ctv_cw_, CTVecCamWorld);          ///< 相机相对世界坐标系原点的位移
  attr_reader_ref(ctv_mw_, CTVecMuzzleWorld);       ///< 枪口相对世界坐标系原点的位移
  attr_reader_ref(rm_ci_, RMatCamIMU);              ///< 相机相对陀螺仪的旋转矩阵
  attr_reader_ref(rm_mi_, RMatMuzzleIMU);           ///< 枪口相对陀螺仪的旋转矩阵
  attr_reader_ref(intrinsic_mat_, IntrinsicMat);    ///< 相机矩阵
  attr_reader_ref(distortion_mat_, DistortionMat);  ///< 畸变矩阵

  bool Initialize();

  /**
   * @brief 将相机坐标系坐标转换为世界坐标系坐标
   * @param [in] ctv_cam 相机坐标系坐标
   * @param [in] rm_imu 当前云台姿态
   * @return 世界坐标系坐标
   */
  [[nodiscard]] CTVec CamToWorld(CTVec REF_IN ctv_cam, RMat REF_IN rm_imu) const;

  /**
   * @brief 将世界坐标系坐标转换为相机坐标系坐标
   * @param [in] ctv_world 世界坐标系坐标
   * @param [in] rm_imu 当前云台姿态
   * @return 相机坐标系坐标
   */
  [[nodiscard]] CTVec WorldToCam(CTVec REF_IN ctv_world, RMat REF_IN rm_imu) const;

  /**
   * @brief 将枪口坐标系坐标转换为世界坐标系坐标
   * @param [in] ctv_muzzle 枪口坐标系坐标
   * @param [in] rm_imu 当前云台姿态
   * @return 世界坐标系坐标
   */
  [[nodiscard]] CTVec MuzzleToWorld(CTVec REF_IN ctv_muzzle, RMat REF_IN rm_imu) const;

  /**
   * @brief 将世界坐标系坐标转换为枪口坐标系坐标
   * @param [in] ctv_world 世界坐标系坐标
   * @param [in] rm_imu 当前云台姿态
   * @return 枪口坐标系坐标
   */
  [[nodiscard]] CTVec WorldToMuzzle(CTVec REF_IN ctv_world, RMat REF_IN rm_imu) const;

  /**
   * @brief 将相机坐标系坐标投影到图像坐标系中
   * @param [in] ctv_cam 相机坐标系坐标
   * @return 图像坐标系点位
   */
  cv::Point2f CamToPic(CTVec REF_IN ctv_cam) const;
};

}  // namespace srm::coord

#endif  // SRM_COORD_SOLVER_H_