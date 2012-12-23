#!/usr/bin/env ruby

# Collectd collector for hadoop namenode stats

require 'getoptlong'
require 'timeout'
require 'syslog'
require 'open3'

if RUBY_VERSION != "1.9.2" then
  puts "We require ruby 1.9.2"
  exit 1
end

$hostname = (ENV['HOSTNAME'] || %x[hostname]).strip

$stdout.sync = true
$from = 'console'

$config = {
  :interval => 60, # seconds
  :iterations => 0,
  :debug => false,
  :sudo => nil,
  :metrics  => {
    :configured_capacity => /Configured Capacity: ([\d]+)/,
    :present_capacity => /Present Capacity: ([\d]+)/,
    :dfs_remaining => /DFS Remaining: ([\d]+)/,
    :dfs_used => /DFS Used: ([\d.]+)/,
    :under_replicated_blocks => /Under replicated blocks: ([\d]+)/,
    :blocks_with_corrupt_replicas => /Blocks with corrupt replicas: ([\d]+)/,
    :missing_blocks => /Missing blocks: ([\d]+)/,
    :datanodes_available => /Datanodes available: ([\d]+)/,
  }
}
opts = GetoptLong.new(
  [ '--debug', '-d', GetoptLong::NO_ARGUMENT ],
  [ '--help', '-h', GetoptLong::NO_ARGUMENT ],
  [ '--interval', '-i', GetoptLong::REQUIRED_ARGUMENT ],
  [ '--iterations', '-I', GetoptLong::REQUIRED_ARGUMENT ],
  [ '--from', '-f', GetoptLong::REQUIRED_ARGUMENT ],
  [ '--sudo', '-s', GetoptLong::REQUIRED_ARGUMENT ],
).each do |opt, arg|
  case opt
  when '--debug'
    $config[:debug] = true
  when '--iterations'
    $config[:iterations] = [arg.to_i, 0].min
  when '--interval'
    $config[:interval] = [arg.to_i, 10].max
  when '--from'
    $from = arg
  when '--sudo'
    $config[:sudo] = arg
  when '--help'
    puts "#{$0} [OPTION]

-d, --debug:
    debug level output

-h, --help:
    show help

-i N, --interval N:
    time to wait between polls, defaults to 60, minimum is 10

-I N, --iterations N:
    number of iterations to do, defaults to indefinite or 10 if from is unspecified

-f S, --from S:
    where this is being run from, defaults to console

-s [U], --sudo [U]:
    user to sudo to, defaults to not using sudo which means this should be run as the appropriate user

Example for dev/prod: #{$0} --debug --interval 10 --sudo hdfs
"
    exit 0
  end
end
# Limit the number of iterations in console mode in case of erroneous func run
if $from == 'console' && $config[:iterations] == 0 then
  $config[:iterations] = 10
end

def log message
  msg = "hdfs_collector - #{$from} - #{message}"
  if $config[:debug] then
    $stderr.puts "hdfs_collector - #{Time.now} - #{message}"
  end
  return if $config[:debug]
  begin
    Timeout::timeout(1) do
      Syslog.open('hdfs_collector', Syslog::LOG_PID | Syslog::LOG_CONS, Syslog::LOG_USER) do |s|
        s.info msg
      end
    end
  rescue Exception => e
    # nowhere to log
  end
end
def log_debug msg
  if $config[:debug] then
    log msg
  end
end

log "Starting"

def shutdown thread, reason
    log "Shutting down due to #{reason}"
    thread.kill
    exit 0
end

def cmd
  if $config[:sudo] && !$config[:sudo].empty? then
    "sudo -u #{$config[:sudo]} hadoop dfsadmin -report"
  else
    "hadoop dfsadmin -report"
  end
end

def get_report_data timeout
  result = []
  log_debug "running command #{cmd}"
  begin
    stdin, stdout, stderr, thread = Open3.popen3(cmd)
  rescue Exception => e
    log "get_report_data - error running #{cmd} - #{e}"
    return result
  end
  log_debug "get_report_data - timeout is #{timeout}"
  begin
    Timeout::timeout(timeout) {
      result = [stdout.read, stderr.read]
    }
  rescue Timeout::Error => e
    log "get_report_data - killing IO thread due to timeout (timeout is #{timeout}), no data received"
    Process.kill('TERM', thread.pid) rescue nil # Do some sloppy process termination
    thread.kill rescue nil # Do some sloppy process termination
  end
  [stdin, stdout, stderr].each(&:close) rescue nil
  result
end

def get_cluster_metrics_from_string s
  s.split("\n\nName:").first
end

def print_metrics ts, report_data
  data = get_cluster_metrics_from_string report_data
  return if data.nil? or data.empty?
  $config[:metrics].sort_by{|k,v| k.to_s}.each do |name,filter|
    value = data[%r[#{filter}], 1]
    next if value.nil? or value.empty?
    puts %Q[PUTVAL "#{$hostname}/exec-hadoop/gauge-#{name}\" interval=#{$config[:interval]} #{ts}:#{value}]
  end
end

thread = Thread.new do

  count = 0
  iterations = $config[:iterations]
  interval = $config[:interval]
  timeout = (interval / 2).to_i

  while true do
    count += 1
    begin
      now = Time.now.to_i
      stdout, stderr = get_report_data(timeout)
      unless stderr.nil? or stderr.empty? then
        log_debug "stderr for command: #{stderr}"
      end
      if stdout then
        print_metrics now, stdout
      end
    rescue Errno::EPIPE => e
      log "collectd process changed and no longer sees my output, goodbye world."
      shutdown Thread.current, 'SUICIDE'
    rescue Timeout::Error => e
      log "DFSADMIN Timed out, failed to obtain hdfs status"
    rescue Exception => e
      log "Exception obtaining hdfs status: #{e}"
      shutdown Thread.current, 'ERROR'
    end
    if iterations > 0 && count > iterations then
      shutdown Thread.current, 'ITERATIONS'
    else
      sleep interval
    end
  end
end

['TERM', 'INT', 'KILL'].each do |s|
  trap(s) do
    shutdown thread, "SIG#{s}"
  end
end

thread.join

