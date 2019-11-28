SHELL := /bin/bash
HIDE:=@
CP = $(HIDE)cp
MKDIR = $(HIDE)mkdir
RM = $(HIDE)rm
MV = $(HIDE)mv

VENV := $(shell pwd)/build
PWD = $(shell pwd)
SRC := ./src
TARGET=$(PWD)/target

## all                    : Compile all the modules
##                          modules: model
all: prep model

## venv                   : Prepare virtualenv
venv:
	$(HIDE)virtualenv -p python3 $(VENV) > /dev/null 2>&1
	$(HIDE)$(VENV)/bin/pip3 install --upgrade pip > /dev/null 2>&1
	$(HIDE)$(VENV)/bin/pip3 install pylint --upgrade > /dev/null 2>&1
	$(HIDE)$(VENV)/bin/pip3 install -r requirements.txt > /dev/null 2>&1

## prep                   : Do the preparation for the compile work
prep:
	$(MKDIR) -p out
	$(MKDIR) -p $(TARGET)
	$(RM) -rf out/*

## src_pylint      : Pylint check for fdb module
src_pylint:
	${HIDE}pushd src > /dev/null 2>&1;$(VENV)/bin/pylint -j4 --rcfile=../pylint.conf --reports=n --output-format=colorized --msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}' *; \
	if [ $$? != 0 ]; then popd; exit -1; fi;popd > /dev/null 2>&1

## model             : Fake database with PII
model: src_pylint
	rm -rf out/*
	cp -rf $(SRC) out/. && \
	python3 -m zipapp out -m "src.model:entry" -o $(TARGET)/model.pyz; \
	rm -rf out/app

help: Makefile
	@sed -n 's/^##//p' $<

## clean                  : Delete all the object files and executables
clean: 
	$(HIDE)find . -name '*.pyc' | xargs rm -f
	$(HIDE)find . -name '*.pyz' | xargs rm -f
	$(HIDE)find . -name '*~' | xargs rm -f
	$(HIDE)find . -name '__pycache__' | xargs rm -rf
	$(RM) -rf ${PROTOAPPDIR}/*pb2.py
	$(RM) -rf out

.PHONY: help all clean prep src_pylint
