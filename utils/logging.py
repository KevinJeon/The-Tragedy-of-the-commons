
def log_statistics_to_writer(logger, step, statistics: dict) -> None:
    logger.log('statistics/movement/move', statistics['movement']['move'], step)
    logger.log('statistics/movement/rotate', statistics['movement']['rotate'], step)
    logger.log('statistics/punishment/punishing', statistics['punishment']['punishing'], step)
    logger.log('statistics/punishment/punished', statistics['punishment']['punished'], step)
    logger.log('statistics/punishment/valid_rate', statistics['punishment']['valid_rate'], step)
    logger.log('statistics/eaten_apples/total/blue', statistics['eaten_apples']['total']['blue'], step)
    logger.log('statistics/eaten_apples/total/red', statistics['eaten_apples']['total']['red'], step)
    logger.log('statistics/eaten_apples/team_blue/blue', statistics['eaten_apples']['team']['blue']['blue'], step)
    logger.log('statistics/eaten_apples/team_blue/red', statistics['eaten_apples']['team']['blue']['red'], step)
    logger.log('statistics/eaten_apples/team_red/red', statistics['eaten_apples']['team']['red']['red'], step)
    logger.log('statistics/eaten_apples/team_red/blue', statistics['eaten_apples']['team']['red']['blue'], step)


def log_agent_to_writer(logger, step, agent_info: dict) -> None:
    for i, info in enumerate(agent_info):
        logger.log('agent_{0}/episode_reward'.format(i), info['accum_reward'], step)
