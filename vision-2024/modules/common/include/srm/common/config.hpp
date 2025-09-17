#ifndef SRM_COMMON_CONFIG_HPP_
#define SRM_COMMON_CONFIG_HPP_

#include <glog/logging.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <toml.hpp>
#include <vector>

#include "srm/common/tags.hpp"

namespace srm {
template <typename T>
concept MatType = std::is_same_v<cv::Mat, T>;
template <typename T>
concept NotMatType = !std::is_same_v<cv::Mat, T>;
/// 命令行参数解析、封装类
class Config final {
 public:
  /**
   * @brief 获取 Config 类唯一实例
   * @return 唯一实例的引用
   */
  static Config &Instance() {
    static Config config;
    return config;
  }

  /**
   * @brief 获取变量的值
   * @tparam T 变量类型,只允许非cv::Mat的类型
   * @param list 变量名
   * @return T 变量值
   * @details
   * 传入的是一个初始化列表，列表中的元素自动用``.``进行拼接，如``cfg.Get<double>({"autoaim.atm"})``的等价写法是``cfg.Get<double>({"autoaim","atm"})``
   */
  template <NotMatType T>
  T Get(std::initializer_list<std::string> FWD_IN list) {
    std::string para_name{};
    for (const auto &it : list) {
      para_name += it + ".";
    }
    para_name.pop_back();
    if (!registry_.contains(para_name)) {
      LOG(ERROR) << para_name << " doesn't exist.";
      return {};
    }
    auto &para = registry_[para_name];
    try {
      auto data = toml::get<T>(para);
      return data;
    } catch (toml::type_error &e) {
      LOG(FATAL) << e.what();
      return {};
    }
  }

  /**
   * @brief 获取变量的值
   * @tparam T 变量类型,只允许cv::Mat的类型
   * @param list 变量名
   * @return T 变量值
   * @details
   * 传入的是一个初始化列表，列表中的元素自动用``.``进行拼接，如``cfg.Get<double>({"autoaim.atm"})``的等价写法是``cfg.Get<double>({"autoaim","atm"})``
   */
  template <MatType T>
  T Get(std::initializer_list<std::string> FWD_IN list) {
    std::string para_name{};
    for (const auto &it : list) {
      para_name += it + ".";
    }
    para_name.pop_back();
    if (!registry_.contains(para_name)) {
      LOG(ERROR) << para_name << " doesn't exist.";
      return {};
    }
    auto &para = registry_[para_name];
    if (!para.is_array()) {
      LOG(FATAL) << para_name << " is not an Matrix.";
      return {};
    }
    if (const auto &array = para.as_array();
        !array.empty() && array[0].is_string() && array[0].as_string() == "opencv-matrix") {
      const auto &table = array[1];
      if (!table.contains("rows") || !table.contains("cols") || !table.contains("data")) {
        LOG(FATAL) << "The format of Matrix " << para_name << " is wrong.";
        return {};
      }
      const int rows = toml::find<int>(table, "rows");
      const int cols = toml::find<int>(table, "cols");
      cv::Mat mat(rows, cols, CV_64F);
      const auto &data = toml::find(table, "data").as_array();
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (auto &value = data[i * cols + j]; value.is_floating()) {
            mat.at<double>(i, j) = value.as_floating();
          } else if (value.is_integer()) {
            mat.at<double>(i, j) = static_cast<double>(value.as_integer());
          } else {
            LOG(FATAL) << "Unsupport element in matrix " << para_name;
            return {};
          }
        }
      }
      return mat;
    }
    LOG(FATAL) << para_name << " is not an Matrix.";
    return {};
  }

  /**
   * @brief 解析配置文件参数
   * @param [in] config_file 配置文件路径
   * @return 是否配置成功
   */
  bool Parse(std::string &&config_file) {
    using namespace std::chrono_literals;
    thread_ = std::make_unique<std::thread>(
        [config_file = std::move(config_file)](Config *self) {
          std::function<void(std::string, toml::value)> dfs = [&](std::string REF_IN name, toml::value REF_IN u) {
            if (u.is_table()) {
              for (const auto &[v_appending_name, v_value] : u.as_table()) {
                std::string v_name = name + (name.empty() ? "" : ".") + v_appending_name;
                dfs(std::move(v_name), v_value);
              }
            } else {
              self->registry_[name] = u;
            }
          };
          const auto config = toml::parse(config_file);
          dfs("", config);
          self->start_flag_ = true;
        },
        this);
    while (!start_flag_) {
      std::this_thread::sleep_for(10ms);
    }
    return true;
  }

 private:
  Config() = default;
  ~Config() {
    stop_flag_ = true;
    if (thread_->joinable()) {
      thread_->join();
    }
  }
  std::unordered_map<std::string, toml::value> registry_{};  ///< 变量的注册表
  std::unique_ptr<std::thread> thread_{};                    ///< 多线程接口
  std::atomic_bool stop_flag_{};                             ///< 退出信号
  std::atomic_bool start_flag_{};                            ///< 配置加载成功信号
  std::mutex registry_lock_{};                               ///< 互斥锁
};
inline Config &cfg = Config::Instance();  ///< 封装命令行参数的全局变量

}  // namespace srm

#endif  // SRM_COMMON_CONFIG_HPP_