#ifndef SRM_COORD_COORD_H_
#define SRM_COORD_COORD_H_

#include <Eigen/Core>

#include "srm/common/tags.hpp"

namespace srm::coord {

// R: Rotation, 旋转
// T: Transition, 位移
// T: Transformation, 变换
// H: Homogeneous, 齐次
// C: Cartesian, 笛卡尔
// S: Spherical, 球面
// E: Euler, 欧拉

using RVec =
    Eigen::Vector3d;  ///< 3x1 旋转向量（以 (x, y, z) 表示，正方向依次为：前移、下移、右移），变量名标记 rv_ 前缀，
using RMat = Eigen::Matrix3d;   ///< 3x3 旋转矩阵，变量名标记 rm_ 前缀
using HTMat = Eigen::Matrix4d;  ///< 4x4 齐次变换矩阵，变量名标记 htm_ 前缀
using CTVec = Eigen::Vector3d;  ///< 3x1 直角坐标位移向量（以 (x, y, z) 表示，正方向依次为：右移、下移、前移）,
                                ///< 变量名标记 ctv_ 前缀
using STVec =
    Eigen::Vector3d;  ///< 3x1 球坐标位移向量（以 (phi, theta, r) 表示，正方向依次为：右偏、上仰）, 变量名标记 stv_ 前缀
using HCTVec = Eigen::Vector4d;  ///< 4x1 齐次位移向量 (x, y, z, 1), 变量名标记 hctv_ 前缀
using EAngle =
    Eigen::Vector3d;  ///< 3x1 欧拉角（以 (yaw, pitch, roll) 表示，正方向依次为：右偏、上仰、右滚），变量名标记 ea_ 前缀

/**
 * @brief 将旋转矩阵转换为欧拉角
 * @param [in] rm 旋转矩阵
 * @return 转换后的欧拉角
 */
EAngle RMatToEAngle(RMat REF_IN rm);

/**
 * @brief 将欧拉角转换为旋转矩阵
 * @param [in] ea 欧拉角
 * @return 转换后的旋转矩阵
 */
RMat EAngleToRMat(EAngle REF_IN ea);

/**
 * @brief 将直角坐标转换为球坐标
 * @param [in] ctv 直角坐标
 * @return 转换后的球坐标
 */
STVec CTVecToSTVec(CTVec REF_IN ctv);

/**
 * @brief 将球坐标转换为直角坐标
 * @param [in] stv 球坐标
 * @return 转换后的直角坐标
 */
CTVec STVecToCTVec(STVec REF_IN stv);

/**
 * @brief 将旋转向量转换为旋转矩阵
 * @param [in] rv 旋转向量
 * @return 转换后的旋转矩阵
 */
RMat RVecToRMat(RVec REF_IN rv);

/**
 * @brief 将旋转矩阵转换为旋转向量
 * @param [in] rm 旋转矩阵
 * @return 转换后的旋转向量
 */
RVec RMatToRVec(RMat REF_IN rm);

}  // namespace srm::coord

#endif  // SRM_COORD_COORD_H_
