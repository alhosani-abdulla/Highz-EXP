# Custom aliases for Highz-EXP analysis environment
# Data analysis and plotting using Python 3.11 with scikit-rf

alias GetSpecs='cd /media/peterson/INDURANCE'
alias AnalysisShell='cd /home/peterson/Highz-EXP && pipenv shell'
export PYTHONPATH=/home/peterson/Highz-EXP/src:$PYTHONPATH
SRC_PATH=/home/peterson/Highz-EXP

ViewSpecs() {
    export PIPENV_PIPFILE=$SRC_PATH/Pipfile
    pipenv run python $SRC_PATH/src/digital_spectrometer/viewspec_now.py "$@"
}

CompressSpecs() {
    export PIPENV_PIPFILE=$SRC_PATH/Pipfile
    pipenv run python $SRC_PATH/src/highz_exp/file_compressor.py "$@"
}

CreatePlot() {
    export PIPENV_PIPFILE=$SRC_PATH/Pipfile
    pipenv run python $SRC_PATH/src/digital_spectrometer/image_creator.py "$@"
}

ViewPlots() {
    export PIPENV_PIPFILE=$SRC_PATH/Pipfile
    pipenv run python $SRC_PATH/src/digital_spectrometer/movie_creator.py "$@"
}
