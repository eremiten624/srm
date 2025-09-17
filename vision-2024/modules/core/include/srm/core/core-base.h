#ifndef SRM_CORE_CORE_BASE_H_
#define SRM_CORE_CORE_BASE_H_

#include "srm/autoaim.hpp"
#include "srm/common.hpp"
#include "srm/coord.hpp"
#include "srm/core/fps-controller.h"
#include "srm/message.hpp"
#include "srm/nn.hpp"
#include "srm/video.hpp"

/// 抓取并处理来自操作系统的控制信号
void SignalHandler(int);

enable_factory(srm::core, BaseCore, CreateCore);

namespace srm::core {

/// 机器人主控公共接口类
class BaseCore {
 protected:
  /// 使SingalHandler能访问exit_signal，以停止程序
  friend void ::SignalHandler(int);
  inline static std::atomic_bool exit_signal = false;  ///< 主循环退出信号

 public:
  BaseCore() = default;
  virtual ~BaseCore() = default;

  /**
   * @brief 初始化机器人
   * @return 是否初始化成功
   */
  virtual bool Initialize();

  /**
   * @brief 执行主控制循环
   * @return 错误码，可作为 main() 的返回值
   */
  virtual int Run() = 0;

 protected:
  video::Frame frame_;                             ///< 帧数据
  std::unique_ptr<video::Reader> reader_;          ///< 视频读入接口
  std::unique_ptr<video::Writer> writer_;          ///< 视频写出接口
  std::shared_ptr<message::BaseMessage> message_;  ///< 串口收发接口
  std::shared_ptr<coord::Solver> solver_;          ///< 坐标求解接口
  std::unique_ptr<FpsController> fps_controller_;  ///< 帧率控制器
  std::shared_ptr<autoaim::BaseAutoaim> autoaim_;  ///< 自瞄接口

  std::unordered_map<autoaim::Mode, std::shared_ptr<autoaim::BaseAutoaim>> autoaim_registry_;  ///< 将模式与自瞄绑定

 private:
  virtual bool InitializeReader();
  virtual bool InitializeWriter();
  virtual bool InitializeMessage();
  virtual bool InitializeSolver();
  virtual bool InitializeAutoaim();
  virtual bool InitializeFpsController();
};

}  // namespace srm::core

#endif  // SRM_CORE_CORE_BASE_H_