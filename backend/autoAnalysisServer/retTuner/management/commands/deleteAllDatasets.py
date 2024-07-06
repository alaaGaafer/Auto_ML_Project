# myapp/management/commands/delete_all_instances.py

from django.core.management.base import BaseCommand
from retTuner.models import datasetsData 

class Command(BaseCommand):
    help = 'Deletes all instances of a specific model.'

    def handle(self, *args, **kwargs):
        datasetsData.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully deleted all instances.'))
