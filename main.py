'''Main python file for running the application'''
from handler import AgentHandler

def main():
    product_name = 'Soundcore Anker R50i'
    AgentHandler().evaluate_product(product_name, type='baseline')
    AgentHandler().evaluate_product(product_name, type='chaining')
    AgentHandler().evaluate_product(product_name, type='parallel')

if __name__ == '__main__':
    main()