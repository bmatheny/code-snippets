# base_template.rb

# Remove some files we will recreate
run 'rm README .gitignore Gemfile'

# Recreate README as a markdown file
file 'README.markdown',
%q{
# New Application: app description

by [Blake Matheny](http://mobocracy.net)
}.strip

# Recreate gitignore file with more useful ignores
file '.gitignore', 
%q{
coverage/*
log/*.log
log/*.pid
db/*.db
db/*.sqlite3
db/schema.rb
tmp/**/*
.DS_Store
doc/api
doc/app
config/database.yml
public/javascripts/all.js
public/stylesheets/all.js
coverage/*
.dotest/*
}.strip

# Recreate the Gemfile with the stuff we need
file 'Gemfile',
%q{
source 'http://rubygems.org'

gem 'rails', '3.0.0'
gem 'sqlite3-ruby', '1.2.5', :require => 'sqlite3'
gem 'haml'

gem 'haml-rails', :git => 'git://github.com/indirect/haml-rails.git'
gem 'rails3-generators', :git => 'git://github.com/indirect/rails3-generators.git'

group :test, :development do
   gem 'rspec-rails', '= 2.0.0.beta.20'
   gem 'webrat'
end

group :test do
   gem 'rspec', '= 2.0.0.beta.20'
   gem 'test-unit', '= 1.2.3'
   gem 'factory_girl'

   gem 'autotest'
   gem 'autotest-rails'

   gem 'autotest-fsevent'
   gem 'autotest-growl'

   gem 'spork'
end
}.strip

run 'bundle install'
generate('rspec:install')


# Need to set to true for production due to Heroku
gsub_file 'config/environments/production.rb', /.*config.serve_static_assets =.*/ , "  config.serve_static_assets = true\n"

# Replace template_engine, test_framework and fixture
gsub_file 'config/application.rb',
	/.*config\.filter_parameters.*/,
%q{
    config.filter_parameters += [:password]
    config.generators do |g|
       g.template_engine :haml
       g.test_framework :rspec, :fixture => true, :views => false
       g.fixture_replacement :factory_girl, :dir => "spec/support/factories"
    end

    # Needed by spork
    if Rails.env.test?
      initializer :after => :initialize_dependency_mechanism do
        ActiveSupport::Dependencies.mechanism = :load
      end
    end
}

# Used by the autotest gem, a little config file
file '.autotest',
%q{
require 'autotest/growl'
require 'autotest/fsevent'

Autotest::Growl::clear_terminal = false
Autotest::Growl::one_notification_per_run = true
}.strip

# Used by the rspec command, a little config file
append_file '.rspec',
%q{
--drb
}

# Needed by factory girl
run 'mkdir -p spec/support/factories'

# Bootstrap haml appropriately
run 'haml --rails .'

# Setup spork to reduce time for tests
append_file 'spec/spec_helper.rb',
%q{
require 'rubygems'
require 'spork'

Spork.prefork do
  ActiveSupport::Dependencies.clear
end

Spork.each_run do
end
}

git :init
git :add => '.'
git :commit => "-a -m 'Initial commit'"
