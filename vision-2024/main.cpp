#include <csignal>

#include "srm/core.hpp"
/**
 * @brief 处理信号函数，用来通知所有线程停止
 */
void SignalHandler(int) {
  LOG(WARNING) << "Caught interrupt signal. Attempting to exit...";
  /// 大概是为了让所有的线程都退出，不要有残留
  srm::core::BaseCore::exit_signal = true;
}

int main(int arc, char** argv) {
  /// 初始化谷歌log
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = FLAGS_colorlogtostderr = FLAGS_log_prefix = true;
  FLAGS_log_dir = "../log/";
  /// 读入配置
  srm::cfg.Parse("../config.toml");
  /// 获取从配置模式
  const auto mode = srm::cfg.Get<std::string>({"mode"});
  /// 工厂造一个core对象，原本用shared_ptr存储是为了传给reader，但是现在传给reader的这个操作已经用lambda捕捉this引用解决了，因此不共享
  /// 因此是unique_ptr就够了
  const std::unique_ptr<srm::core::BaseCore> core(srm::core::CreateCore(mode));
  /// 未创建成功
  if (!core) {
    LOG(ERROR) << "Failed to create " << mode << " core";
    return -1;
  }
  /// 创建成功
  if (!core->Initialize()) {
    LOG(ERROR) << "Failed to initialize core.";
    return 1;
  }
  std::signal(SIGINT, &SignalHandler);  ///< 当Ctrl+C的时候触发信号
  std::signal(SIGTERM, &SignalHandler);  ///< 当有终止信号的时候...，终止信号一般是由操作系统发来的
  /// 正式运行程序
  const int ret = core->Run();
  /// 关闭google log
  google::ShutdownGoogleLogging();
  /// 以错误码退出
  return ret;
}