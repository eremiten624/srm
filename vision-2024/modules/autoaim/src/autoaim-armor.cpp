#include "srm/autoaim/autoaim-armor.h"

#include "srm/autoaim/drawer.h"

namespace srm::autoaim {

bool ArmorAutoaim::Initialize() {
  /// 公共初始化
  bool ret = BaseAutoaim::Initialize();
  /// 初始化detector,chooser,processor
  armor_detector_ = std::make_unique<ArmorDetector>();
  ret &= armor_detector_->Initialize();

  if (!ret) {
    LOG(ERROR) << "Failed to initialize autoaim for armor.";
    return false;
  }
  return true;
}

bool ArmorAutoaim::Run() {
  ArmorPtrList armor_list_;
  /// 完成识别和处理，最终需要得到yaw_和pitch_的数据，注意这两个数据并不是相对角，而是要根据电控传来的rm_self_来进行计算其绝对角
  /// 如果未识别到，请发送0，这个时候机器人会自动进行视野的扫描，但如果想要自行在没识别到的时候也要自己操纵机器人的方向，也可不赋值为0
#ifdef DEBUG
  viewer_->SendFrame(image_);
#endif

  return true;
}
bool ArmorAutoaim::InitializeViewerImpl() {
  return viewer_->Initialize(cfg.Get<std::string>({"viewer.web.shm_name", "armor"}),
                             cfg.Get<int>({"viewer.web", "armor"}));
}

}  // namespace srm::autoaim
