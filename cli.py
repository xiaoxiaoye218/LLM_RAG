#命令行接口，执行后可以在命令行窗口与程序交互
import yaml, argparse
from executor import MilvusExecutor
from easydict import EasyDict

def read_yaml_config(path):
    with open(path,"r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

class CommandLine():
    def __init__(self,config_path):
        self._mode = None
        self._executor = None
        self.config_path = config_path

    def show_start_info(self):
        with open('./start_info.txt') as stif:
            print(stif.read())

    def run(self):
        self.show_start_info()
        config = read_yaml_config(self.config_path)
        self._executor = MilvusExecutor(config)
        print('(rag) milvus模式已选择')
        print('  1.使用`build data/history_24/baihuasanguozhi.txt`来进行知识库构建。')
        print('  2.已有索引可以使用`ask`进行提问, `-d`参数以debug模式进入。')
        print('  3.删除已有索引可以使用`remove baihuasanguozhi.txt`。')

        while True:
            command_text = input("(rag) ")
            self.parse_input(command_text)

    #对输入进行解析，并根据解析执行动作
    def parse_input(self,text):
        commands = text.split(' ')
        if commands[0] == "build":
            if len(commands)==3:
                if commands[1] == "-overwrite":
                    print(commands)
                    self.build_index(path=commands[2],overwrite=True)
                else:
                    print("(rag) build仅支持'-overwrite'参数")
            elif len(commands)==2:
                self.build_index(path=commands[1],overwrite=False)
        elif commands[0] == "ask":
            if len(commands) == 2:
                if commands[1] == "-d":
                    self._executor.set_debug(True)
                else:
                    print("(rag) ask仅支持'-d'参数")
            else:
                self._executor.set_debug(False)
            self.question_answer()                  #进入ask模式
        elif commands[0] == "remove":
            if len(commands)!=2:
                print("(rag) remove仅支持1个参数（文件名|-all)")
            else:
                self._executor.delete_file(commands[1])
        elif commands[0] == "quit":
            self._exit()
        else:
            print("(rag) 只有[build|ask|remove|quit]中的操作，请重新输入")

    def query(self,question):
        ans = self._executor.query(question)
        print(ans)
        print('+---------------------------------------------------------------------------------------------------------------------+')
        print('\n')

        with open('QA.txt', 'a', encoding='utf-8') as file:
            file.write(f"问题: {question}\n")
            file.write(f"回答: {ans}\n")
            file.write('+---------------------------------------------------------------------------------------------------------------------+\n\n')

    def build_index(self,path,overwrite):
        self._executor.build_index(path,overwrite)
        print('(rag) 索引构建完毕')

    def remove(self,filename):
        self._executor.delete_file(filename)

    #在每次进入ask模式时调用
    def question_answer(self):
        self._executor.build_query_engine()
        while True:
            question = input('(rag) 请输入问题： ')
            if question == "quit":
                print("(rag) 退出ask模式")
                break
            elif question == "":
                continue
            self.query(question)

    def _exit(self):
        exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()      #把parser挂到命令行上
    parser.add_argument('--cfg',type=str,help='Path to the configuration file',default='cfgs/config.yaml')
    args = parser.parse_args()              #解析命令行传入的参数，并将结果存储在 args 对象中

    cli = CommandLine(args.cfg)
    cli.run()
