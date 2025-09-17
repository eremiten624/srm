#ifndef SRM_MESSAGE_MESSAGE_BASE_HPP_
#define SRM_MESSAGE_MESSAGE_BASE_HPP_

#include <glog/logging.h>

#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

#include "srm/common.hpp"
#include "srm/message/packet.hpp"

enable_factory(srm::message, BaseMessage, CreateMessage);

namespace srm::message {

/// 数据包通信类
class BaseMessage {
 public:
  BaseMessage() = default;
  virtual ~BaseMessage() = default;

  /**
   * @brief 初始化通信模块
   * @param type 模式(物理/模拟)
   * @return 是否初始化成功
   */
  virtual bool Initialize() { return false; };

  /**
   * @brief 发送缓冲区的数据到目标
   * @warning 调用该函数会清空缓冲区，一般由主控来调用
   * @return 是否发送成功
   */
  virtual bool Send() { return false; }

  /**
   * @brief 接目标数据到缓冲区
   * @warning 调用该函数会覆盖缓冲区，一般由主控来调用
   * @return 是否接收成功
   */
  virtual bool Receive() { return false; }

  /**
   * @brief 与目标建立或断开连接
   * @param flag 为真时表示建立连接，为假时表示断开连接
   * @return 操作是否成功
   */
  virtual bool Connect(bool flag) { return false; };

  /**
   * @brief 为接收数据包分配编号
   * @tparam T 数据包类型
   * @param id 分配的编号
   * @return 是否分配成功
   */
  template <typename T>
  bool ReceiveRegister(short id);

  /**
   * @brief 为发送数据包分配编号
   * @tparam T 数据包类型
   * @param id 分配的编号
   * @return 是否分配成功
   */
  template <typename T>
  bool SendRegister(short id);

  /**
   * @brief 从缓冲区中读取带编号的数据
   * @param [out] data 传出的数据
   * @return 是否读取成功
   */
  template <typename T>
  bool ReadData(T REF_OUT data);

  /**
   * @brief 写入带编号的数据到缓冲区
   * @param [in] data 传入的数据
   * @return 是否写入成功
   */
  template <typename T>
  bool WriteData(T REF_IN data);

 protected:
  using Registry = std::unordered_map<std::string, std::pair<short, short>>;  //(typename,(id,size))，类型有唯一id

  template <typename T>
  bool Register(Registry REF_OUT registry, short id);

  template <typename T>
  short GetId(Registry REF_IN registry) const;

  Registry receive_registry_{};                        ///< 接收数据包注册表
  Registry send_registry_{};                           ///< 发送数据包注册表
  short receive_size_{};                               ///< 接收大小
  Packet receive_buffer_{};                            ///< 接收缓冲区
  short send_size_{};                                  ///< 发送大小
  Packet send_buffer_{};                               ///< 发送缓冲区
  std::unordered_map<short, Packet> packet_received_;  ///< 存储拆分好的接收数据包: (id,data)，每一个Packet就是一个数据
};

template <typename T>
bool BaseMessage::ReceiveRegister(const short id) {
  if (!Register<T>(receive_registry_, id)) {
    LOG(ERROR) << "Failed to register receive packet.";
    return false;
  }
  packet_received_[id].Resize(sizeof(T));  /// 设定拆分后的数据的大小
  receive_size_ += sizeof(short) /*id大小*/ + sizeof(T);
  return true;
}

template <typename T>
bool BaseMessage::SendRegister(const short id) {
  if (!Register<T>(send_registry_, id)) {
    LOG(ERROR) << "Failed to register send packet.";
    return false;
  }
  send_size_ += sizeof(short) + sizeof(T);
  return true;
}

template <typename T>
bool BaseMessage::ReadData(T REF_OUT data) {
  const auto id = GetId<T>(receive_registry_);
  if (!packet_received_.contains(id)) {
    LOG(ERROR) << "Can not find packet in received data.";
    return false;
  }
  Packet packet = packet_received_[id];
  packet.Read(data);  /// 解小包
  return true;
}

template <typename T>
bool BaseMessage::WriteData(T REF_IN data) {
  /// 只能写已经注册过类型对应id的对象
  const short id = GetId<T>(send_registry_);
  if (!id) {
    LOG(ERROR) << "Unregistered packet";
    return false;
  }
  send_buffer_.Write(id);
  send_buffer_.Write(data);
  return true;
}

template <typename T>
bool BaseMessage::Register(Registry REF_OUT registry, short id) {
  /// 检查id和类型是否都没注册
  const std::string name = typeid(T).name();
  for (const auto& [id_, size_] : registry | std::views::values) {
    if (id_ == id) {
      LOG(ERROR) << "ID " << id << " is already in use.";
      return false;
    }
  }
  if (registry.contains(name)) {
    LOG(ERROR) << "Type " << name << " is already registered.";
    return false;
  }
  /// 注册
  registry[name] = {id, sizeof(T)};
  return true;
}

template <typename T>
short BaseMessage::GetId(Registry REF_IN registry) const {
  /// 获取对应id
  const std::string name = typeid(T).name();
  if (registry.contains(name)) {
    return registry.at(name).first;
  }
  LOG(ERROR) << "Packet is not registered.";
  return 0;
}

}  // namespace srm::message

#endif  // SRM_MESSAGE_MESSAGE_BASE_HPP_