from env import TOCEnv

def main():
    env = TOCEnv()
    state = env.reset()
    print('partial', state)

    full_state = env.get_full_state()
    print('full', full_state)


if __name__ == '__main__':
    main()