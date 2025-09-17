#ifndef SRM_AUTOAIM_AUTOAIM_ARMOR_H_
#define SRM_AUTOAIM_AUTOAIM_ARMOR_H_

#include <memory>

#include "srm/autoaim/autoaim-base.h"
#include "srm/autoaim/detector-armor.h"

namespace srm::autoaim {

// 装甲板自瞄类
class ArmorAutoaim final : public BaseAutoaim {
  inline static auto registry = RegistrySub<BaseAutoaim, ArmorAutoaim>("armor");

 public:
  friend class Drawer;
  bool Initialize() override;
  bool Run() override;

  bool InitializeViewerImpl() override;

 private:
  std::unique_ptr<ArmorDetector> armor_detector_;      ///< 装甲板识别器
};

}  // namespace srm::autoaim

#endif  // SRM_AUTOAIM_AUTOAIM_ARMOR_H_