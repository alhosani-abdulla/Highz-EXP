# Custom aliases
alias TakeSpecs='python ~/Highz-EXP/src/digital_spectrometer/run_spectrometer.py'
alias GetSpecs='cd /media/peterson/INDURANCE'

ViewSpecs() {
    if [[ "$VIRTUAL_ENV" != *"postprocess"* ]]; then
        if [[ -n "$VIRTUAL_ENV" ]]; then
            deactivate
        fi
        source ~/postprocess/bin/activate
    fi
    python ~/Highz-EXP/src/digital_spectrometer/viewspec_now.py "$@"
}

export PYTHONPATH=~/Highz-EXP/src:$PYTHONPATH

CreatePlot() {
    if [[ "$VIRTUAL_ENV" != *"postprocess"* ]]; then
        if [[ -n "$VIRTUAL_ENV" ]]; then
            deactivate
        fi
        source ~/postprocess/bin/activate
    fi
    python ~/Highz-EXP/src/digital_spectrometer/image_creator.py "$@"
}

ViewPlots() {
    if [[ "$VIRTUAL_ENV" != *"postprocess"* ]]; then
        if [[ -n "$VIRTUAL_ENV" ]]; then
            deactivate
        fi
        source ~/postprocess/bin/activate
    fi
    python ~/Highz-EXP/src/digital_spectrometer/movie_creator.py "$@"
}
