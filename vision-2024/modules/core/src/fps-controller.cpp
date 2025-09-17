#include "srm/core/fps-controller.h"

#include <iostream>
#include <thread>

namespace srm::core {

using std::chrono_literals::operator""ms;

void FpsController::Tick() {
  while (static_cast<double>((clock::now() - last_time_).count()) < frame_interval_) {
    std::this_thread::sleep_for(1ms);
  }

  const auto current_time = clock::now();
  actual_fps_ = 1e9 / static_cast<double>((current_time - last_time_).count());
  last_time_ = current_time;
}

}  // namespace srm::core