# This can be executed in the hbase shell and will create a table named stats. Stats has
# 4 column families all using LZO compression. Additionally this pre-splits the regions, but assumes
# keys are alpha-numeric
splits = (('a'..'z').to_a + ['-','_'] + ('A'..'Z').to_a + (0..9).to_a.map{|i| i.to_s})
create 'stats',
  {NAME => 'data', COMPRESSION => 'LZO', VERSIONS => 1},
  {NAME => 'std', COMPRESSION => 'LZO',  VERSIONS => 1, TTL => 604800},
  {NAME => 'stw', COMPRESSION => 'LZO', VERSIONS => 1, TTL => 2419200},
  {NAME => 'stm', COMPRESSION => 'LZO', VERSIONS => 1, TTL => 31449600},
  {SPLITS => splits}
