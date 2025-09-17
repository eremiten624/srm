#ifndef SRM_MESSAGE_INFO_H_
#define SRM_MESSAGE_INFO_H_
namespace srm::message {

/// 发送的云台数据
struct GimbalSend {
  float yaw;    ///< 绝对yaw角度
  float pitch;  ///< 绝对pitch角度
};

/// 接收的云台数据
struct GimbalReceive {
  float yaw;
  float pitch;
  float roll;
  int mode;
  int color;
};

/// 发送给打弹的数据
struct ShootSend {
  int fire_flag;
};

/// 接收的打弹数据
struct ShootReceive {
  float bullet_speed;  ///< 弹速
};

/// 合并的接收数据
struct ReiceivePacket {
  float yaw;
  float pitch;
  float roll;
  int mode;
  int color;
  float bullet_speed;
};

}  // namespace srm::message

#endif
