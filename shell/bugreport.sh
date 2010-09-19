#!/bin/sh
PATH=/sbin:/usr/sbin:/bin:/usr/bin:$PATH
DATE=`date +%m-%d-%Y-%s`
FILE=sysinfo.$DATE

TEST=`type type|grep not`
if test -n "$TEST"; then
  WHICH=which
else
  WHICH=type
fi

echo "Software information" >> $FILE
echo "--------------------------------------------------------------" >> $FILE
uname -s 2>/dev/null |\
	awk '{print "OS             :",$N}' >> $FILE

uname -r 2>/dev/null |\
	awk '{print "OS RELEASE     :",$N}' >> $FILE

uname -m 2>/dev/null |\
	awk '{print "HARDWARE       :",$N}' >> $FILE

uname -a 2>/dev/null |\
	awk '{print "UNAME          :",$N}' >> $FILE

TEST=`$WHICH ls 2>/dev/null`
if test -n "$TEST"; then
  ls -l `ldd /bin/sh | awk '/libc/{print $3}'` |\
    sed -e 's/\.so$//' | awk -F'[.-]'   '{print "GNU C Library  : " \
    $(NF-2)"."$(NF-1)"."$NF}' >> $FILE
else
  echo "ls             : Not Found" >> $FILE
fi

TEST=`$WHICH gcc 2>/dev/null`
if test -n "$TEST"; then
  gcc --version 2>/dev/null |\
	  head -1 |\
	  awk '{print "gcc version    :",$N}' >> $FILE
else
  echo "gcc version    : Not Found" >> $FILE
fi

TEST=`$WHICH gmake 2>/dev/null`
if test -n "$TEST" ; then
	gmake --version 2>/dev/null |\
		awk -F, '{print $1}' |\
		awk '/GNU Make/{print "Gnu gmake      :",$NF}' >> $FILE
else
  TEST=`make --version 2>/dev/null`
  if test -n "$TEST"; then
	make --version 2>/dev/null |\
		awk -F, '{print $1}' |\
		awk '/GNU Make/{print "Gnu make       :",$NF}' >> $FILE
  else
	echo "Gnu Make       : Not Found" >> $FILE
  fi
fi

TEST=`$WHICH ld 2>/dev/null`
if test -n "$TEST"; then
  ld --version 2>/dev/null |\
    head -1 |\
    awk '{print "ld             : "$4}' >> $FILE
else
  echo "ld             : Not Found" >> $FILE
fi

TEST=`$WHICH ldd 2>/dev/null`
if test -n "$TEST"; then
  ldd -v >/dev/null 2>&1 && ldd -v || ldd --version | head -1 |\
    awk 'NR==1{print "ldd            :", $NF}' >> $FILE
else
  echo "ldd            : Not Found" >> $FILE
fi

TEST=`$WHICH cp 2>/dev/null`
if test -n "$TEST"; then
  cp --version 2>/dev/null |\
    head -1 |\
    awk '{print "coreutils      : "$4}' >> $FILE
else
  echo "coreutils      : Not Found" >> $FILE
fi

TEST=`$WHICH libtool 2>/dev/null`
if test -n "$TEST"; then
  libtool --version |\
    head -1 |\
    awk '{\
	if (length($4) == 0) {\
		print "libtool        : "$3\
	} else {\
		print "libtool        : "$4\
	}}' >> $FILE
else
  echo "libtool        : Not Found" >> $FILE
fi

TEST=`$WHICH autoconf 2>/dev/null`
if test -n "$TEST"; then
  autoconf --version |\
    head -1 |\
    awk '{\
	if (length($4) == 0) {\
		print "autoconf       : "$3\
	} else {\
		print "autoconf       : "$4\
	}}' >> $FILE
else
  echo "autoconf       : Not Found" >> $FILE
fi

TEST=`$WHICH automake 2>/dev/null`
if test -n "$TEST"; then
  automake --version 2>/dev/null |\
    head -1 |\
    awk '{print "automake       : "$4}' >> $FILE
else
  echo "automake       : Not Found" >> $FILE
fi

echo "--------------------------------------------------------------" >> $FILE

echo "" >> $FILE
echo "CPU Info" >> $FILE
echo "-----------------------------------------------------------" >> $FILE
cat /proc/cpuinfo >> $FILE

echo "" >> $FILE
echo "Module Info" >> $FILE
echo "-----------------------------------------------------------" >> $FILE
cat /proc/modules >> $FILE

echo "" >> $FILE
echo "Ioports Info" >> $FILE
echo "-----------------------------------------------------------" >> $FILE
cat /proc/ioports >> $FILE

echo "" >> $FILE
echo "Iomem Info" >> $FILE
echo "-----------------------------------------------------------" >> $FILE
cat /proc/iomem >> $FILE

echo "" >> $FILE
echo "lspci Info" >> $FILE
echo "-----------------------------------------------------------" >> $FILE
lspci -vvv >> $FILE

echo "$FILE contains information which may help resolve your"
echo "bug, please submit it with your bug report."
