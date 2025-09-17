#include "srm/autoaim/drawer.h"

#include <fmt/format.h>

#include <numeric>
#include <opencv2/imgproc.hpp>

#include "srm/autoaim/info.hpp"

namespace srm::autoaim {

void Drawer::DrawArmor(ArmorPtr REF_IN armor) const {
  std::stringstream armor_info;
  // armor_info << color_id_map_.at(armor->color);
  auto& image = autoaim_->image_;
  const auto& points = armor->pts;
  for (int j = 0; j < 4; j++) {
    line(image, points[j], points[(j + 1) % 4], kGreen, 1);
  }
  const auto center = std::accumulate(points.begin(), points.end(), cv::Point2f(0, 0));
  circle(image, center, 2, kGreen, 2);
}

void Drawer::DrawWorldPoint(coord::CTVec REF_IN ctv_w_origin_x) const {
  const auto& solver = autoaim_->coord_solver_;
  const auto& rm_self = autoaim_->rm_self_;
  const cv::Point2f point_p_target_x = solver->CamToPic(solver->WorldToCam(ctv_w_origin_x, rm_self));
  circle(autoaim_->image_, point_p_target_x, 2, kGreen, 2);
}

}  // namespace srm::autoaim
