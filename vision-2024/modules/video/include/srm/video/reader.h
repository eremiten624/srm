#ifndef SRM_VIDEO_READER_H_
#define SRM_VIDEO_READER_H_

#include <atomic>
#include <opencv2/videoio.hpp>

#include "srm/common/config.hpp"
#include "srm/video/camera.h"
#include "srm/video/frame.hpp"

enable_factory(srm::video, Reader, CreateReader);

namespace srm::video {
/// 视频源公共接口类
class Reader {
 public:
  Reader() = default;
  virtual ~Reader() = default;

  /**
   * @brief 初始化视频源
   * @param [in] prefix 前置路径
   * @return 是否初始化成功
   */
  virtual bool Initialize(std::string REF_IN prefix) = 0;

  /**
   * @brief 获取帧数据
   * @param [out] frame 帧数据输出
   * @return 是否获取成功
   */
  virtual bool GetFrame(Frame REF_OUT frame) = 0;

  /**
   * @brief 注册帧数据回调函数
   * @param [in] callback 回调函数
   */
  virtual void RegisterFrameCallback(FrameCallback FWD_IN callback) = 0;

  /**
   * @brief 读取相机针孔模型矩阵
   * @return const cv::Mat& 相机针孔模型矩阵
   */
  [[nodiscard]] virtual const cv::Mat &IntrinsicMat() const = 0;

  /**
   * @brief 读取相机畸变矩阵
   * @return const cv::Mat& 相机畸变矩阵
   */
  [[nodiscard]] virtual const cv::Mat &DistortionMat() const = 0;

 protected:
  int source_count_{};  ///< 视频源数量
};
}  // namespace srm::video

#endif  // SRM_VIDEO_READER_H_
