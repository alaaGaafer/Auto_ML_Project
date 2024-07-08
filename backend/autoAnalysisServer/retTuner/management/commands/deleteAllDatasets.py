# myapp/management/commands/delete_all_instances.py

from django.core.management.base import BaseCommand
from retTuner.models import datasetsData,usersData
import os

class Command(BaseCommand):
    help = 'Deletes all instances of a specific model.'

    def handle(self, *args, **kwargs):
        datasetsData.objects.all().delete()
        usersData.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully deleted all instances.'))
        paths = ['preprocessing_Scripts/media/','preprocessing_Scripts/models/']
        for media_directory in paths:
            print(os.listdir(media_directory))
            for filename in os.listdir(media_directory):

                file_path = os.path.join(media_directory, filename)
                try:
                        os.remove(file_path)
                        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {file_path}.'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Failed to delete {file_path}. Reason: {str(e)}'))

            self.stdout.write(self.style.SUCCESS('Successfully deleted all files in the media directory.'))
