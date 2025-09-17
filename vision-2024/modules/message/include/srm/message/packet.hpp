#ifndef SRM_MESSAGE_PACKET_HPP_
#define SRM_MESSAGE_PACKET_HPP_

#include <memory>
#include <vector>

#include "srm/common.hpp"

namespace srm::message {

/**
 * @brief 字节流数据包类
 * @details
 * 采用网络通信中的首尾定界法，为了能够正常读取，必须要知道每个数据的大小，也就是类型，因此要给每个数据一个id，这个id将会和类型绑定
 */
class Packet : std::vector<char> {
 public:
  template <typename T>
  char* change(T* ptr) {
    return reinterpret_cast<char*>(const_cast<std::add_pointer_t<std::remove_cv_t<T>>>(ptr));
  }
  /**
   * @brief 写入数据
   * @param [in] data 要写入的数据
   * @return 是否写入成功
   */
  template <typename T>
  bool Write(T REF_IN data) {
    insert(this->end(), change(&data), change(&data) + sizeof(data));
    return true;
  }

  /**
   * @brief 读取数据
   * @param [out] data 传出读取的数据
   * @return 是否读取成功
   */
  template <typename T>
  bool Read(T REF_OUT data) {
    /// 如果越界
    if (this->begin() + read_offset_ + sizeof(data) > this->end()) {
      return false;
    }
    /// 不知道换成这样会不会出问题，感觉应该不会出
    /// 内存复制数据
    std::copy_n(this->begin() + read_offset_, sizeof(data), change(&data));
    read_offset_ += sizeof(data);
    return true;
  }

  /// 大写，未覆盖
  void Clear() {
    this->clear();
    read_offset_ = 0;
  }

  /**
   * @brief 重新调整包大小
   * @param n 新的包大小
   * @warning read_offset_也会被重置
   */
  void Resize(const size_t n) {
    this->resize(n);
    read_offset_ = 0;
  }

  [[nodiscard]] size_t Size() const { return this->size(); }
  [[nodiscard]] bool Empty() const { return read_offset_ >= this->size(); }
  [[nodiscard]] char const* Ptr() const { return this->data(); };
  char* Ptr() { return this->data(); };

 private:
  long read_offset_{};
};

};  // namespace srm::message

#endif  // SRM_MESSAGE_PACKET_HPP_