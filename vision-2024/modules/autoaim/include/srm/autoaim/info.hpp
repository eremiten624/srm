#ifndef SRM_AUTOAIM_INFO_HPP_
#define SRM_AUTOAIM_INFO_HPP_

#include <numeric>
#include <opencv2/core/mat.hpp>

#include "srm/coord/coord-base.h"

namespace srm::autoaim {

/// 自瞄模式
enum class Mode {
  kArmor = 0,      ///< 本次校内赛模式
  kSmallRune = 1,
  kBigRune = 2,
};

/// 颜色
enum class Color {
  kBlue = 0,
  kRed = 1,
  kGrey = 2,
  kPurple = 3,
};
const auto kGreen = cv::Scalar(0, 192, 0);
const auto kRed = cv::Scalar(0, 0, 192);
const auto kBlue = cv::Scalar(192, 0, 0);
const auto kPurple = cv::Scalar(192, 0, 192);
const auto kYellow = cv::Scalar(0, 192, 192);

struct Armor {
  std::array<cv::Point2f, 4> pts;  ///< 装甲板在图片中的角点信息
  Color color;                     ///< 装甲板颜色
  coord::RMat rm_cam;              ///< 装甲板旋转矩阵
  coord::CTVec ctv_w_x;            ///< 装甲板位移向量

  Armor(const std::array<cv::Point2f, 4> &pts, const Color color) : pts(pts), color(color) {}

  [[nodiscard]] cv::Point2f Center() const { return std::accumulate(pts.begin(), pts.end(), cv::Point2f(0, 0))/4; }
};
using ArmorPtr = std::shared_ptr<Armor>;
using ArmorPtrList = std::vector<ArmorPtr>;

}  // namespace srm::autoaim

#endif  // SRM_AUTOAIM_INFO_HPP_
