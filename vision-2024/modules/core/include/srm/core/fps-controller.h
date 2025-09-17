#ifndef SRM_CORE_FPS_
#define SRM_CORE_FPS_

#include <chrono>
#include <srm/common/tags.hpp>

namespace srm::core {
using clock = std::chrono::high_resolution_clock;
/// 运行帧率控制类
class FpsController final {
 public:
  /// 将initialize更改为了一个构造函数，感觉真没必要分离
  explicit FpsController(const double target_fps) : frame_interval_(1e9 / target_fps), last_time_(clock::now()) {}
  ~FpsController() = default;

  /**
   * @brief 控制函数
   * @details 在循环内加入该语句来限制帧率
   */
  void Tick();

  /// 获取实际的帧数
  attr_reader_val(actual_fps_, GetFPS);

 private:
  double frame_interval_{};                                                ///< 要求的每一帧之间的间隔
  double actual_fps_{};                                                    ///< 运行时实际帧率
  std::chrono::time_point<std::chrono::high_resolution_clock> last_time_;  ///< 上一次tick的时间
};

}  // namespace srm::core
#endif  // SRM_CORE_FPS_
