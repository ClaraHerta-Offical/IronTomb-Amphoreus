import pstats

# 性能分析工具
p = pstats.Stats('profile.stats')
p.strip_dirs().sort_stats('cumulative').print_stats(20)