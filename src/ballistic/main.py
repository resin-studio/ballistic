import util

from textx import metamodel_from_file
parser = metamodel_from_file(util.resource('grammars/bll.tx'))

def parse_from_file(path):
    return parser.model_from_file(path)

if __name__ == "__main__":
    program = parse_from_file(util.resource('examples/hello.bll'))
    print('-----------------------------------------------------')
    print("Greeting", ", ".join([to_greet.name for to_greet in program.to_greet]))
    print('-----------------------------------------------------')
    # print(f'''
    # -------------------------
    # {util.resource("bll.tx")}
    # -------------------------
    # {util.project_path()}
    # -------------------------
    # ''')
    # pass