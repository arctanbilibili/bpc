# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = E:\CMake\bin\cmake.exe

# The command to remove a file.
RM = E:\CMake\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Administrator\Desktop\bpc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Administrator\Desktop\bpc\build

# Include any dependencies generated for this target.
include CMakeFiles/bbb.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bbb.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bbb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bbb.dir/flags.make

CMakeFiles/bbb.dir/bpc.c.obj: CMakeFiles/bbb.dir/flags.make
CMakeFiles/bbb.dir/bpc.c.obj: ../bpc.c
CMakeFiles/bbb.dir/bpc.c.obj: CMakeFiles/bbb.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Administrator\Desktop\bpc\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/bbb.dir/bpc.c.obj"
	C:\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/bbb.dir/bpc.c.obj -MF CMakeFiles\bbb.dir\bpc.c.obj.d -o CMakeFiles\bbb.dir\bpc.c.obj -c C:\Users\Administrator\Desktop\bpc\bpc.c

CMakeFiles/bbb.dir/bpc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bbb.dir/bpc.c.i"
	C:\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Administrator\Desktop\bpc\bpc.c > CMakeFiles\bbb.dir\bpc.c.i

CMakeFiles/bbb.dir/bpc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bbb.dir/bpc.c.s"
	C:\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Administrator\Desktop\bpc\bpc.c -o CMakeFiles\bbb.dir\bpc.c.s

# Object files for target bbb
bbb_OBJECTS = \
"CMakeFiles/bbb.dir/bpc.c.obj"

# External object files for target bbb
bbb_EXTERNAL_OBJECTS =

bbb.exe: CMakeFiles/bbb.dir/bpc.c.obj
bbb.exe: CMakeFiles/bbb.dir/build.make
bbb.exe: CMakeFiles/bbb.dir/linklibs.rsp
bbb.exe: CMakeFiles/bbb.dir/objects1.rsp
bbb.exe: CMakeFiles/bbb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Administrator\Desktop\bpc\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bbb.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\bbb.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bbb.dir/build: bbb.exe
.PHONY : CMakeFiles/bbb.dir/build

CMakeFiles/bbb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\bbb.dir\cmake_clean.cmake
.PHONY : CMakeFiles/bbb.dir/clean

CMakeFiles/bbb.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\Administrator\Desktop\bpc C:\Users\Administrator\Desktop\bpc C:\Users\Administrator\Desktop\bpc\build C:\Users\Administrator\Desktop\bpc\build C:\Users\Administrator\Desktop\bpc\build\CMakeFiles\bbb.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bbb.dir/depend

