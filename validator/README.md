# Submission Format Validation Tool

Both provided excutable file named in `submission_validator` can help you to validate your submission format.

## Usage

Please make sure that your submission file is named with your student username (s1234567.csv) and in the same directory as the `submission_validator` file. Then, double click the `submission_validator` file to run it (exe for Windows, no suffix for mac OS and linux).

Once the tool has been executed, the following instructions will be shown:

```plaintext
--------------------------------------------------
          Submission Validator
--------------------------------------------------

This tool can help you to validate your submission format.

Please place your submission file in the same directory as this tool, and type your student username (s1234567), or type "exit" to quit:
```

Then, type your student username (e.g. s1234567) and press enter to start the validation. If you want to quit, type `exit` and press enter.

## Potential Issues

You may encounter security issues when running the tool. If so, please follow the instructions below to solve the problem.

### Windows

Windows may block the execution by displaying a security warning. If so, please click `More info` and then `Run anyway` to continue.

### Mac OS

Mac OS may block the execution as this tool has no developer signature. If so, please open your terminal change the directory to the folder where the tool is located, and run the following command to allocate the permission to execute the tool:

```bash
chmod +x submission_validator
```

## Output

If your submission format is correct, you will see the following output:

```bash
Validating s1234567 ...

Passed 	 check file existence passed
Passed 	 check length passed
Passed 	 check delimiter passed
Passed 	 check lines passed
Passed 	 check report line passed

Done.
```

Otherwise, you will be informed with the error message:

```bash
Validating s7654321 ...

Passed 	 check file existence passed
Passed 	 check length passed
Failed 	 Each line must be comma-separated.
Failed 	 Each prediction line must be composed in the format of "SINGLE_DIGIT,\n"
Passed 	 check report line passed

Done.
```

Try the provided `s1234567.csv` and `s7654321.csv` to take a quick look at the output.
