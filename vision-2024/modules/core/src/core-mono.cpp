#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "srm/core.hpp"

namespace srm::core {

/**
 * @brief 常规机器人主控类
 * @warning 禁止直接构造此类，请使用 @code srm::core::CreateCore("normal") @endcode 获取该类的公共接口指针
 */
class NormalCore final : public BaseCore {
  inline static auto registry = RegistrySub<BaseCore, NormalCore>("normal");  ///< 主控注册信息
 public:
  int Run() override;

 private:
  bool UpdateFrameList();
  bool SetAutoaim();
  void SendData() const;
};

int NormalCore::Run() {
  while (!exit_signal) {
    fps_controller_->Tick();
    LOG_EVERY_N(INFO, 100) << fps_controller_->GetFPS();
    if (!UpdateFrameList()) {
      continue;
    }
    if (!SetAutoaim()) {
      continue;
    }
    autoaim_->Run();
    if (message_) {
      SendData();
    }
  }
  return 0;
}

bool NormalCore::UpdateFrameList() {
  static bool show_warning = true;
  static int warning_count = 0;
  const auto ret = reader_->GetFrame(frame_);
  if (!ret && show_warning) {
    LOG(WARNING) << "Waiting to get valid data.";
    warning_count++;
  } else if (ret && !show_warning) {
    warning_count = 0;
    LOG(WARNING) << "Problem of waiting to get valid data has been solved.";
  } else if (warning_count == 25) {
    warning_count = 0;
    LOG(WARNING) << "It seems that it has been a long time not receiving a valid data, it may be the problem of the "
                    "config setting, please check the hardware trigger setting.";
  }
  show_warning = ret;
  return ret;
}

bool NormalCore::SetAutoaim() {
  static bool show_warning = true;
  const bool ret = frame_.sync_data != nullptr;
  if (!ret && show_warning) {
    LOG(WARNING) << "No sync data found.";
  } else if (ret && !show_warning) {
    LOG(WARNING) << "The problem of no sync data found has been solved.";
  }
  show_warning = ret;
  if (!ret) {
    return false;
  }

  const auto receive_packet = std::static_pointer_cast<message::ReiceivePacket>(frame_.sync_data);
  const auto& [yaw, pitch, roll, mode_int, color_int, bullet_speed] = *receive_packet;
  const auto mode = static_cast<autoaim::Mode>(mode_int);
  const auto color = static_cast<autoaim::Color>(color_int);
  if (!autoaim_registry_.contains(mode)) {
    LOG(ERROR) << "Unknown mode for autoaim.";
    return false;
  }
  autoaim_ = autoaim_registry_[mode];
  autoaim_->SetMode(mode);
  autoaim_->SetColor(color);
  autoaim_->SetBulletSpeed(bullet_speed);
  autoaim_->SetTimeStamp(frame_.time_stamp);
  autoaim_->SetImageList(frame_.image);

  const coord::EAngle ea_self = {yaw, pitch, roll};
  autoaim_->SetRmSelf(coord::EAngleToRMat(ea_self));

  return true;
}

void NormalCore::SendData() const {
  const message::GimbalSend gimbal_send{autoaim_->GetYaw(), autoaim_->GetPitch()};
  const message::ShootSend shoot_send{autoaim_->IsFire()};
  message_->WriteData(gimbal_send);
  message_->WriteData(shoot_send);
  message_->Send();
}

}  // namespace srm::core