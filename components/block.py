class BlockType:
    Empty = 1 << 0
    OutBound = 1 << 1
    # Wall = 2
    Apple = 1 << 2

    #  Agents
    Self = 1 << 3
    Others = 1 << 4

    #  Effects
    Punish = 1 << 20

    # Numbering should be up to 63 (uint64)






