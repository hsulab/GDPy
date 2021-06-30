#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def run_command(directory, command, comment=''):
    proc = subprocess.Popen(
        command, shell=True, cwd=directory, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding = 'utf-8'
    )
    errorcode = proc.wait(timeout=120) # 10 seconds
    if errorcode:
        raise ValueError('Error in %s at %s.' %(comment, directory))
    
    return ''.join(proc.stdout.readlines())