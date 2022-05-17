import warnings

from steps.human_evaluation import understandability

__EXECUTION_DICT = {
    1: understandability.export_clusters_sample_images
}

__MENU = """
exit: terminate the program
0: Execute all

1: Export the data for the understandability human study

Select one or more of the options separated by a space: """

__INVALID_OPTION = lambda option: f'Invalid option [{option}]'

if __name__ == '__main__':
    # Add the choice for execute all
    __EXECUTION_DICT[0] = __EXECUTION_DICT.values()
    warnings.filterwarnings('ignore')
    choices_str = input(__MENU)
    while choices_str != 'exit':
        try:
            choices = [int(choice) for choice in choices_str.split(' ')]

            # Get the handler for the input
            for choice in choices:
                handler = __EXECUTION_DICT.get(choice)
                if handler is not None:
                    try:
                        handler()
                    except TypeError:
                        [handler_item() for handler_item in handler]
                else:
                    print(__INVALID_OPTION(choices_str))
        except ValueError:
            print(__INVALID_OPTION(choices_str))

        choices_str = input(__MENU)
