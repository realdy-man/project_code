import socket

def get_local_ip():
    try:
        # 创建一个 UDP 连接（不会真正发送数据）
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"
# # 使用
# host = get_local_ip()
# register_to_nacos(SERVICE_NAME, host, SERVICE_PORT)