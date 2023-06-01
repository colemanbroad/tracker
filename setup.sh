printf "\nHere's the path ... \n\n"
printf $PATH
printf "\n\nZig we're using is...\n\n"
which zig
printf "\n"
# export PATH=$PATH

PATH=""
PATH+=":/bin"
PATH+=":/sbin"
PATH+=":/usr/bin"
PATH+=":/usr/sbin"
PATH+=":/usr/local/bin"
PATH+=":/usr/local/sbin"
# PATH+=":/Library/TeX/texbin"
# PATH+=":/Users/broaddus/.nimble/bin"
PATH+=":/Users/broaddus/Desktop/software-thirdparty/zig-macos-x86_64-0.10.0-dev.2033+3679d737f" ## note! this goes before ~/bin to use zig ?
PATH+=":/Users/broaddus/bin"
export PATH

printf "Here's the new path ... \n\n"
printf $PATH
printf "\n"


# alias zig=""/Users/broaddus/Desktop/software-thirdparty/zig-macos-x86_64-0.10.0-dev.2033+3679d737f/zig"