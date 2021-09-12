class BlockType:
    Empty = 1 << 0
    OutBound = 1 << 1
    # Wall = 2
    RedApple = 1 << 2
    BlueApple = 1 << 3

    #  Agents
    Self = 1 << 4

    # RedAgent = 1 << 5
    BlueAgent = 1 << 6
    GreenAgent = 1 << 7
    OrangeAgent = 1 << 8
    PurpleAgent = 1 << 9

    #  Effects
    Punish = 1 << 20

    # Numbering should be up to 63 (uint64)






