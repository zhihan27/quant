"""通过服务器终端管理授权账号，不提供公开注册入口。"""

import argparse
import getpass

from services.auth import AuthService
from services.database import DatabaseError, QuantRepository


def main() -> int:
    """解析账号管理命令，密码从终端读取，避免进入命令历史。"""
    parser = argparse.ArgumentParser(description="量化工作台账号管理")
    parser.add_argument("action", choices=["create", "reset", "disable"], help="创建、重置密码或禁用账号")
    parser.add_argument("username", help="账号名")
    parser.add_argument("--temporary", action="store_true", help="设为临时密码，首次登录强制修改")
    args = parser.parse_args()
    try:
        service = AuthService(QuantRepository())
        if args.action == "disable":
            service.disable_user(args.username)
        else:
            password = getpass.getpass("请输入密码：")
            repeated = getpass.getpass("请再次输入密码：")
            if password != repeated:
                raise ValueError("两次输入的密码不一致。")
            if args.action == "create":
                service.create_user(args.username, password, args.temporary)
            else:
                service.set_password(args.username, password, args.temporary)
        print("账号操作完成。")
        return 0
    except (DatabaseError, ValueError) as exc:
        print(f"操作失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
