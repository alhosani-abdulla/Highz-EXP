# Custom aliases
alias TakeSpecs='python ~/Highz-EXP/src/digital_spectrometer/run_spectrometer.py'
alias ViewSpecs='python ~/Desktop/RTV_nosave_V4.py'
alias GetSpecs='cd /media/peterson/INDURANCE'

export PATH=~/Highz-EXP/src:$PATH

CreatePlot() {
    python ~/Highz-EXP/src/digital_spectrometer/image_creator.py "$@"
}