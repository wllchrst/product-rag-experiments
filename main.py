'''Main python file for running the application'''
from handler import AgentHandler

def main():
    answer = AgentHandler().test_evaluation_agent()
    print(answer)
    
if __name__ == '__main__':
    main()