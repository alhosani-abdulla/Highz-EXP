# Custom aliases for Highz-EXP analysis environment
# Data analysis and plotting using Python 3.11 with scikit-rf

alias GetSpecs='cd /home/peterson/Data/INDURANCE'
alias AnalysisShell='cd /home/peterson/Highz-EXP && pipenv shell'

ViewSpecs() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/digital_spectrometer/viewspec_now.py "$@"
}

CompressSpecs() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/highz_exp/file_compressor.py "$@"
}

export PYTHONPATH=/home/peterson/Highz-EXP/src:$PYTHONPATH

CreatePlot() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/digital_spectrometer/image_creator.py "$@"
}

ViewPlots() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/digital_spectrometer/movie_creator.py "$@"
}
