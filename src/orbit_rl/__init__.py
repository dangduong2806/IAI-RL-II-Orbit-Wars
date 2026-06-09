"""Orbit Wars PPO Mission Selector v2.

Pipeline:
obs -> lb-1200 WorldModel -> generate top-K missions -> PPO selects one mission
-> lb-1200 executes selected mission -> env.step -> reward -> PPO update.
"""
