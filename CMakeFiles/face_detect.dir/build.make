# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jean-romain/Documents/face_recognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jean-romain/Documents/face_recognition

# Include any dependencies generated for this target.
include CMakeFiles/face_detect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/face_detect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/face_detect.dir/flags.make

CMakeFiles/face_detect.dir/src/main.cpp.o: CMakeFiles/face_detect.dir/flags.make
CMakeFiles/face_detect.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jean-romain/Documents/face_recognition/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/face_detect.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_detect.dir/src/main.cpp.o -c /home/jean-romain/Documents/face_recognition/src/main.cpp

CMakeFiles/face_detect.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_detect.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jean-romain/Documents/face_recognition/src/main.cpp > CMakeFiles/face_detect.dir/src/main.cpp.i

CMakeFiles/face_detect.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_detect.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jean-romain/Documents/face_recognition/src/main.cpp -o CMakeFiles/face_detect.dir/src/main.cpp.s

CMakeFiles/face_detect.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/face_detect.dir/src/main.cpp.o.requires

CMakeFiles/face_detect.dir/src/main.cpp.o.provides: CMakeFiles/face_detect.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_detect.dir/build.make CMakeFiles/face_detect.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/face_detect.dir/src/main.cpp.o.provides

CMakeFiles/face_detect.dir/src/main.cpp.o.provides.build: CMakeFiles/face_detect.dir/src/main.cpp.o


CMakeFiles/face_detect.dir/src/faceArray.cpp.o: CMakeFiles/face_detect.dir/flags.make
CMakeFiles/face_detect.dir/src/faceArray.cpp.o: src/faceArray.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jean-romain/Documents/face_recognition/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/face_detect.dir/src/faceArray.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_detect.dir/src/faceArray.cpp.o -c /home/jean-romain/Documents/face_recognition/src/faceArray.cpp

CMakeFiles/face_detect.dir/src/faceArray.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_detect.dir/src/faceArray.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jean-romain/Documents/face_recognition/src/faceArray.cpp > CMakeFiles/face_detect.dir/src/faceArray.cpp.i

CMakeFiles/face_detect.dir/src/faceArray.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_detect.dir/src/faceArray.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jean-romain/Documents/face_recognition/src/faceArray.cpp -o CMakeFiles/face_detect.dir/src/faceArray.cpp.s

CMakeFiles/face_detect.dir/src/faceArray.cpp.o.requires:

.PHONY : CMakeFiles/face_detect.dir/src/faceArray.cpp.o.requires

CMakeFiles/face_detect.dir/src/faceArray.cpp.o.provides: CMakeFiles/face_detect.dir/src/faceArray.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_detect.dir/build.make CMakeFiles/face_detect.dir/src/faceArray.cpp.o.provides.build
.PHONY : CMakeFiles/face_detect.dir/src/faceArray.cpp.o.provides

CMakeFiles/face_detect.dir/src/faceArray.cpp.o.provides.build: CMakeFiles/face_detect.dir/src/faceArray.cpp.o


# Object files for target face_detect
face_detect_OBJECTS = \
"CMakeFiles/face_detect.dir/src/main.cpp.o" \
"CMakeFiles/face_detect.dir/src/faceArray.cpp.o"

# External object files for target face_detect
face_detect_EXTERNAL_OBJECTS =

face_detect: CMakeFiles/face_detect.dir/src/main.cpp.o
face_detect: CMakeFiles/face_detect.dir/src/faceArray.cpp.o
face_detect: CMakeFiles/face_detect.dir/build.make
face_detect: /usr/local/lib/libopencv_dnn.so.3.4.3
face_detect: /usr/local/lib/libopencv_ml.so.3.4.3
face_detect: /usr/local/lib/libopencv_objdetect.so.3.4.3
face_detect: /usr/local/lib/libopencv_shape.so.3.4.3
face_detect: /usr/local/lib/libopencv_stitching.so.3.4.3
face_detect: /usr/local/lib/libopencv_superres.so.3.4.3
face_detect: /usr/local/lib/libopencv_videostab.so.3.4.3
face_detect: /usr/local/lib/libopencv_calib3d.so.3.4.3
face_detect: /usr/local/lib/libopencv_features2d.so.3.4.3
face_detect: /usr/local/lib/libopencv_flann.so.3.4.3
face_detect: /usr/local/lib/libopencv_highgui.so.3.4.3
face_detect: /usr/local/lib/libopencv_photo.so.3.4.3
face_detect: /usr/local/lib/libopencv_video.so.3.4.3
face_detect: /usr/local/lib/libopencv_videoio.so.3.4.3
face_detect: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
face_detect: /usr/local/lib/libopencv_imgproc.so.3.4.3
face_detect: /usr/local/lib/libopencv_core.so.3.4.3
face_detect: CMakeFiles/face_detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jean-romain/Documents/face_recognition/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable face_detect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/face_detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/face_detect.dir/build: face_detect

.PHONY : CMakeFiles/face_detect.dir/build

CMakeFiles/face_detect.dir/requires: CMakeFiles/face_detect.dir/src/main.cpp.o.requires
CMakeFiles/face_detect.dir/requires: CMakeFiles/face_detect.dir/src/faceArray.cpp.o.requires

.PHONY : CMakeFiles/face_detect.dir/requires

CMakeFiles/face_detect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/face_detect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/face_detect.dir/clean

CMakeFiles/face_detect.dir/depend:
	cd /home/jean-romain/Documents/face_recognition && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jean-romain/Documents/face_recognition /home/jean-romain/Documents/face_recognition /home/jean-romain/Documents/face_recognition /home/jean-romain/Documents/face_recognition /home/jean-romain/Documents/face_recognition/CMakeFiles/face_detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/face_detect.dir/depend

