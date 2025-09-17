#include "srm/autoaim.hpp"

namespace srm::autoaim {

bool BaseAutoaim::Initialize() {
#ifdef DEBUG
  return InitializeViewer() && InitializeDrawer();
#else
  return InitializeDrawer();
#endif
}

bool BaseAutoaim::InitializeViewer() {
  /// 现在只能web了，没有local了
  viewer_ = std::make_unique<viewer::VideoViewer>();
  if (!InitializeViewerImpl()) {
    LOG(ERROR) << "Failed to initialize viewer.";
    return false;
  }
  LOG(INFO) << "Viewer is initialized successfully.";
  return true;
}

bool BaseAutoaim::InitializeDrawer() {
  drawer_ = std::make_shared<Drawer>();
  drawer_->InitializeAutoaim(this);
  return true;
}

}  // namespace srm::autoaim