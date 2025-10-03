# Custom aliases
alias TakeSpecs='python ~/Highz-EXP/src/digital_spectrometer/run_spectrometer.py'
alias ViewSpecs='python ~/Desktop/RTV_nosave_V4.py'
alias GetSpecs='cd /media/peterson/INDURANCE'
alias TestView='python ~/Highz-EXP/src/digital_spectrometer/viewspec_now.py'

export PYTHONPATH=~/Highz-EXP/src:$PYTHONPATH

CreatePlot() {
    python ~/Highz-EXP/src/digital_spectrometer/image_creator.py "$@"
}
