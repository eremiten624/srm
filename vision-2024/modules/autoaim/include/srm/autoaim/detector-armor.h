#ifndef SRM_AUTOAIM_DETECTOR_ARMOR_H_
#define SRM_AUTOAIM_DETECTOR_ARMOR_H_

#include "srm/autoaim/info.hpp"
#include "srm/common.hpp"
#include "srm/coord.hpp"
#include "srm/nn.hpp"
#include "srm/viewer.hpp"

namespace srm::autoaim {

/// 装甲板识别器
class ArmorDetector {
 public:
  attr_writer_val(coord_solver_, InitCoordSolver);

  bool Initialize();

  /**
   * @brief 运行装甲板检测器
   * @param [in] image 传入图片
   * @param [out] armor_list 传出装甲板列表
   * @return 是否运行成功
   */
  bool Run(cv::Mat REF_IN image, ArmorPtrList REF_OUT armor_list)const;

 private:
  std::unique_ptr<nn::Yolo> yolo_;               ///< 神经网络接口
  std::shared_ptr<coord::Solver> coord_solver_;  ///< 坐标求解器接口
  std::shared_ptr<viewer::VideoViewer> viewer_;  ///< 可视化接口
};

}  // namespace srm::autoaim

#endif  // SRM_AUTOAIM_DETECTOR_ARMOR_H_