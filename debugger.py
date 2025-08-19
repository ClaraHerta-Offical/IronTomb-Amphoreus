import os
import sys
import threading
import time

try:
    # Windows
    import msvcrt
except ImportError:
    # Unix-like
    import tty, termios, select

from constants import BAIE_STAGNATION_THRESHOLD, PATH_NAMES, TITAN_NAMES

class Debugger:
    def __init__(self, simulation):
        self.sim = simulation
        self.paused = False
        self.last_command = '' 
        self._keyboard_thread = threading.Thread(target=self._listen_for_keys, daemon=True)
        self._keyboard_thread.start()

    def _listen_for_keys(self):
        """监听键盘输入的独立线程，仅用于触发暂停。"""
        while True:
            # 持续检查是否有 'p' 键被按下以切换暂停状态
            char = self._get_char()
            if char == 'p':
                self.paused = not self.paused
                # 给主循环一点时间来响应暂停状态的改变
                time.sleep(0.1) 
            time.sleep(0.05)

    def _get_char(self):
        """跨平台的非阻塞获取单个字符的函数。"""
        if os.name == 'nt':
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').lower()
                except UnicodeDecodeError:
                    return None
        else:
            # 保存终端原始设置
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                # 设置终端为原始模式
                tty.setraw(sys.stdin.fileno())
                # 检查是否有输入
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    return sys.stdin.read(1).lower()
            finally:
                # 恢复终端设置
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None

    def handle_commands(self):
        """在模拟暂停时处理用户输入的命令。"""
        while self.paused:
            try:
                command_line_str = input("\033[93m(翁法罗斯创世涡心) > \033[0m").strip()

                if not command_line_str: continue
                
                command_line = command_line_str.split()
                cmd = command_line[0].lower()
                args = command_line[1:]

                if cmd in ('c', 'continue'):
                    self.paused = False
                    break # 退出命令处理循环，主循环将恢复
                elif cmd in ('n', 'next'):
                    self.last_command = 'next'
                    self.paused = False # 执行一帧然后暂停
                    break
                elif cmd in ('p', 'print'):
                    if not args:
                        print("错误: 请提供实体名称或 'baie'。用法: p <name|baie>")
                        continue
                    entity_name = ' '.join(args)
                    entity = None
                    if entity_name.lower() == 'baie':
                        entity = self.sim.reincarnator
                        if not entity: print("错误: 当前没有卡厄斯兰那实体。"); continue
                    else:
                        entity = self.sim.name_to_entity_map.get(entity_name)
                    
                    if entity:
                        print(f"\n--- 实体详情: {entity.name} ---")
                        print(entity)
                        print("  泰坦亲和度:")
                        for i, name in enumerate(TITAN_NAMES):
                            print(f"    {name:<4}: {entity.titan_affinities[i]:.2f}")
                        print("  命途倾向:")
                        for i, name in enumerate(PATH_NAMES):
                            print(f"    {name:<4}: {entity.path_affinities[i]:.3f}")
                        print("---")
                    else:
                        print(f"错误: 未找到名为 '{entity_name}' 的实体。")

                elif cmd == 'top':
                    k = int(args[0]) if args and args[0].isdigit() else 5
                    sorted_pop = sorted(self.sim.population, key=lambda p:p.score, reverse=True)
                    top_k = sorted_pop[:k]
                    print(f"\n--- 当前评分 Top {k} ---")
                    for i, p in enumerate(top_k): print(f"{i+1}. {p}")
                    print("---")
                elif cmd == 'status':
                    diversity = 0
                    if self.sim.population:
                       diversity = len(set(p.dominant_path_idx for p in self.sim.population)) / len(PATH_NAMES)
                    print("\n--- 翁法罗斯状态报告 ---")
                    print(f"  世代: {self.sim.generation}/{self.sim.total_generations}")
                    print(f"  种群数量: {len(self.sim.population)}")
                    print(f"  生态多样性: {diversity:.2%}")
                    print(f"  当前突变率: {self.sim.stagnation_manager.mutation_rate:.4f}")
                    if self.sim.stagnation_manager.long_term_stagnation_counter:
                        print(f"  全局停滞计数: {self.sim.stagnation_manager.long_term_stagnation_counter} / 10 (触发唤醒)")
                    if self.sim.reincarnator:
                        print(f"  白厄停滞计数: {self.sim.stagnation_manager.baie_stagnation_counter} / {BAIE_STAGNATION_THRESHOLD}")
                    print("---")
                elif cmd == 'zeitgeist':
                    print("\n--- 当前翁法罗斯思潮 ---")
                    zeitgeist_status = sorted(zip(PATH_NAMES, self.sim.cosmic_zeitgeist), key=lambda item: item[1], reverse=True)
                    for name, weight in zeitgeist_status:
                        print(f"  {name:<4}: {weight:+.4f}")
                    print("---")
                elif cmd == 'blueprint':
                    print("\n--- 当前演化蓝图 ---")
                    blueprint_status = sorted(zip(TITAN_NAMES, self.sim.base_titan_affinities), key=lambda item: item[1], reverse=True)
                    for name, affinity in blueprint_status:
                        print(f"  {name:<4}: {affinity:.4f}")
                    print("---")
                elif cmd == 'set':
                    if len(args) != 2: print("错误: 用法: set <parameter_name> <value>"); continue
                    param, value = args[0], args[1]
                    if hasattr(self.sim, param):
                        try:
                            current_val = getattr(self.sim, param)
                            setattr(self.sim, param, type(current_val)(value))
                            print(f"成功: 参数 '{param}' 已被设置为 {value}。")
                        except (ValueError, TypeError):
                            print(f"错误: 无法将 '{value}' 转换为 '{param}' 所需的类型。")
                    else: print(f"错误: 模拟中不存在名为 '{param}' 的参数。")

                elif cmd == 'save':
                    if not args:
                        print("错误: 请提供存档文件名。用法: save <filename.json>")
                        continue
                    filepath = args[0]
                    self.sim.save_simulation_state(filepath)

                elif cmd == 'load':
                    if not args:
                        print("错误: 请提供要加载的存档文件名。用法: load <filename.json>")
                        continue
                    filepath = args[0]
                    self.sim.load_simulation_state(filepath)
                    print("状态已加载。输入 'c' 或 'n' 继续。")

                elif cmd == 'help':
                    print("\n--- 可用命令 ---")
                    print("  c, continue         : 继续模拟")
                    print("  n, next             : 执行下一世代并暂停")
                    print("  p, print <name|baie>: 打印指定实体或当前卡厄斯兰那的详细信息")
                    print("  top [k]             : 显示评分最高的k个实体 (默认 k=5)")
                    print("  status              : 显示当前的翁法罗斯宏观状态")
                    print("  zeitgeist           : 查看当前的翁法罗斯思潮权重")
                    print("  blueprint           : 查看当前的演化蓝图亲和度")
                    print("  set <param> <value> : 动态设置一个模拟参数 (如: set mutation_rate 0.5)")
                    print("  save <file.json>    : 将当前模拟状态保存到文件")
                    print("  load <file.json>    : 从文件加载模拟状态")
                    print("  help                : 显示此帮助信息")
                    print("---")
                else:
                    print(f"错误: 未知命令 '{cmd}'。")

            except (KeyboardInterrupt, EOFError):
                print("\n强制恢复模拟...")
                self.paused = False
                break
