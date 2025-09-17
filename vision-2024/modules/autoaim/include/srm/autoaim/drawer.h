#ifndef SRM_AUTOAIM_DRAWER_H_
#define SRM_AUTOAIM_DRAWER_H_

#include "srm/autoaim/autoaim-base.h"
#include "srm/autoaim/info.hpp"
#include "srm/coord/coord-solver.h"

namespace srm::autoaim {

class BaseAutoaim;

/// 绘图类
class Drawer {
 public:
  Drawer() = default;
  ~Drawer() = default;

  attr_writer_val(autoaim_, InitializeAutoaim);

  /// 绘制单装甲板
  void DrawArmor(ArmorPtr REF_IN armor) const;

  /// 根据世界坐标在图像中进行绘制
  void DrawWorldPoint(coord::CTVec REF_IN ctv_w_origin_x) const;

 protected:
  BaseAutoaim* autoaim_;

 private:
  static inline const std::unordered_map<Color, std::string> color_id_map_ = {
      {Color::kBlue, "B"}, {Color::kRed, "R"}, {Color::kGrey, "G"}, {Color::kPurple, "P"}};
};

}  // namespace srm::autoaim

#endif