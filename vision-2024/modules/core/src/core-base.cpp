#include <memory>

#include "srm/core.hpp"

namespace srm::core {

bool BaseCore::InitializeReader() {
  const auto type = cfg.Get<std::string>({"type"});
  const auto reader_type = cfg.Get<std::string>({"video.reader"});
  reader_.reset(video::CreateReader(reader_type));
  if (!reader_) {
    LOG(ERROR) << "Failed to create " << reader_type << " reader.";
    return false;
  }
  if (!reader_->Initialize("video." + type + "." + reader_type)) {
    LOG(ERROR) << "Failed to initialize reader.";
    return false;
  }
  while (!reader_->GetFrame(frame_)) {
    LOG(WARNING) << "Waiting for the first frame frome video.";
    sleep(1);
  }
  LOG(INFO) << "Reader is initialized successfully.";
  return true;
}

bool BaseCore::InitializeWriter() {
  if (!reader_) {
    return false;
  }
  if (!cfg.Get<bool>({"video.writer"})) {
    LOG(INFO) << "Writer do not need initialization.";
    return true;
  }
  writer_ = std::make_unique<video::Writer>();
  using std::chrono::system_clock;
  const auto t_str = std::format("{:%Y-%m-%d-%H.%M.%S}", system_clock::now());
  if (!writer_->Open(std::format("../cache/{}-{}.avi", cfg.Get<std::string>({"type"}), t_str),
                     {frame_.image.cols, frame_.image.rows})) {
    LOG(ERROR) << "Failed to open video file. Please check your disk space.";
    return false;
  }
  LOG(INFO) << "Writer is initialized successfully.";
  return true;
}

bool BaseCore::InitializeMessage() {
  if (!reader_) {
    return false;
  }
  if (cfg.Get<bool>({"control"})) {
    message_.reset(message::CreateMessage("control"));
    if (!message_->Initialize()) {
      LOG(ERROR) << "Failed to open shared memory communication.";
      return false;
    }
    std::string prefix = "message.control";
    message_->ReceiveRegister<message::GimbalReceive>(cfg.Get<short>({prefix, "receive.gimbal"}));
    message_->ReceiveRegister<message::ShootReceive>(cfg.Get<short>({prefix, "receive.shoot"}));
    message_->SendRegister<message::GimbalSend>(cfg.Get<short>({prefix, "send.gimbal"}));
    message_->SendRegister<message::ShootSend>(cfg.Get<short>({prefix, "send.shoot"}));
    message_->Connect(true);
  }
  auto func = std::make_unique<video::FrameCallback::function>([this](video::Frame &frame) {
    auto *receive_packet = new message::ReiceivePacket();
    if (message_) {
      message_->Receive();
      message::GimbalReceive gimbal_receive{};
      message::ShootReceive shoot_receive{};
      if (!message_->ReadData(gimbal_receive) || !message_->ReadData(shoot_receive)) {
        LOG(WARNING) << "Failed to read data in frame callback function. Set this frame as invalid.";
        frame.valid = false;
      }
      receive_packet->yaw = gimbal_receive.yaw;
      receive_packet->pitch = gimbal_receive.pitch;
      receive_packet->roll = gimbal_receive.roll;
      receive_packet->mode = gimbal_receive.mode;
      receive_packet->color = gimbal_receive.color;
      receive_packet->bullet_speed = shoot_receive.bullet_speed;
    } else {
      const std::string prefix = "message.simulator";
      receive_packet->yaw = cfg.Get<float>({prefix, "yaw"});
      receive_packet->pitch = cfg.Get<float>({prefix, "pitch"});
      receive_packet->roll = cfg.Get<float>({prefix, "roll"});
      receive_packet->bullet_speed = cfg.Get<float>({prefix, "bullet_speed"});
      receive_packet->mode = cfg.Get<int>({prefix, "mode"});
      receive_packet->color = cfg.Get<int>({prefix, "color"});
    }
    frame.sync_data.reset(receive_packet);
  });
  reader_->RegisterFrameCallback({std::move(func)});
  LOG(INFO) << "Serial is initialized successfully.";
  return true;
}

bool BaseCore::InitializeSolver() {
  if (!reader_) {
    return false;
  }
  solver_ = std::make_shared<coord::Solver>();
  solver_->InitIntrinsicMat(reader_->IntrinsicMat());
  solver_->InitDistortionMat(reader_->DistortionMat());
  if (!solver_->Initialize()) {
    LOG(ERROR) << "Failed to initialize coordinate solver.";
    return false;
  }
  if (cfg.Get<bool>({"coord", cfg.Get<std::string>({"type"}), "cam_flip"})) {
    auto func = std::make_unique<video::FrameCallback::function>([](video::Frame &frame) {
      flip(frame.image, frame.image, 0);
      flip(frame.image, frame.image, 1);
    });
    reader_->RegisterFrameCallback({std::move(func)});
  }

  LOG(INFO) << "Solver is initialized successfully.";
  return true;
}

bool BaseCore::InitializeAutoaim() {
  if (!solver_) {
    return false;
  }
  auto &autoaim = autoaim_registry_[autoaim::Mode::kArmor];
  autoaim.reset(autoaim::CreateAutoaim("armor"));
  if (!autoaim) {
    LOG(ERROR) << "Failed to create armor-autoaim.";
    return false;
  }
  autoaim->InitCoordSolver(solver_);
  if (!autoaim->Initialize()) {
    LOG(ERROR) << "Failed to initialize autoaim.";
    return false;
  }

  LOG(INFO) << "Autoaim is initialized successfully.";
  return true;
}

bool BaseCore::InitializeFpsController() {
  fps_controller_ = std::make_unique<FpsController>(cfg.Get<double>({"fps_limit"}));
  return true;
}

bool BaseCore::Initialize() {
  bool ret = true;

  ret &= InitializeReader();
  if (!ret) {
    LOG(ERROR) << "Failed to initialize base core because of Reader part.";
    return false;
  }

  ret &= InitializeWriter();
  ret &= InitializeMessage();
  ret &= InitializeSolver();
  ret &= InitializeFpsController();
  if (!ret) {
    LOG(ERROR) << "Failed to initialize base core because of Writer | Message | Solver | FpsController part.";
    return false;
  }

  ret &= InitializeAutoaim();
  if (!ret) {
    LOG(ERROR) << "Failed to initialize base core because of autoaim part.";
    return false;
  }

  LOG(INFO) << "Successfully initialized base environment of " << cfg.Get<std::string>({"type"}) << " controller.";
  return true;
}

}  // namespace srm::core