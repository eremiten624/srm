#ifndef SRM_AUTOAIM_AUTOAIM_BASE_H_
#define SRM_AUTOAIM_AUTOAIM_BASE_H_

#include <opencv2/core/mat.hpp>

#include "srm/autoaim/info.hpp"
#include "srm/common.hpp"
#include "srm/coord.hpp"
#include "srm/nn.hpp"
#include "srm/viewer.hpp"

enable_factory(srm::autoaim, BaseAutoaim, CreateAutoaim);

namespace srm::autoaim {

class Drawer;
/// 自瞄类
class BaseAutoaim {
 public:
  friend class Drawer;
  BaseAutoaim() = default;
  virtual ~BaseAutoaim() = default;

  attr_writer_val(coord_solver_, InitCoordSolver);

  attr_writer_val(image_, SetImageList);
  attr_writer_val(time_stamp_, SetTimeStamp);
  attr_writer_val(rm_self_, SetRmSelf);
  attr_writer_val(bullet_speed_, SetBulletSpeed);
  attr_writer_val(mode_, SetMode);
  attr_writer_val(color_, SetColor);

  attr_reader_ref(yaw_, GetYaw);
  attr_reader_ref(pitch_, GetPitch);
  attr_reader_ref(fire_, IsFire);

  /**
   * @brief 初始化自瞄
   * @return 是否初始化成功
   */
  virtual bool Initialize();

  /**
   * @brief 运行自瞄
   * @return 当前检测帧是否有效
   */
  virtual bool Run() { return false; };

 protected:
  std::shared_ptr<coord::Solver> coord_solver_;        ///< 坐标求解器
  std::shared_ptr<Drawer> drawer_;                     ///< 绘图类
  std::unique_ptr<viewer::VideoViewer> viewer_;        ///< 图像显示接口

  // 传入的参数
  cv::Mat image_{};        ///< 图片
  uint64_t time_stamp_{};  ///< 时间戳
  coord::RMat rm_self_{};  ///< 位姿矩阵
  float bullet_speed_{};   ///< 弹丸速度
  Mode mode_{};            ///< 自瞄模式
  Color color_{};          ///< 自身颜色

  // 传出的参数
  float yaw_{};    ///< 水平方向
  float pitch_{};  ///< 竖直方向
  bool fire_{};    ///< 是否开火

  /// 初始化图像显示接口
  virtual bool InitializeViewer();

  /// 初始化绘图类
  virtual bool InitializeDrawer();

  ///真正调用viewer_->initialize的函数
  virtual bool InitializeViewerImpl()=0;
};

}  // namespace srm::autoaim

#endif  // SRM_AUTOAIM_AUTOAIM_BASE_H_